"""g2 `discourse` generator (spec section 4.5): deterministic, no llm.

walks atoms, starting a new "region" at any `pause_before_ms >= 1500` (or a
speaker change -- a documented no-op in phase 1, see below); within each
region, emits every contiguous atom run whose span fits
`[cfg.min_duration_ms, cfg.max_duration_ms]`, capped at 4 candidates per
region.
"""

from __future__ import annotations

from clipper.config import Config
from clipper.generators.base import build_candidate
from clipper.types import Atom, Candidate

_REGION_PAUSE_THRESHOLD_MS = 1500
_MAX_CANDIDATES_PER_REGION = 4


def _split_into_regions(atoms: list[Atom]) -> list[list[Atom]]:
    """split `atoms` into regions: a new region starts at any atom whose
    `pause_before_ms >= _REGION_PAUSE_THRESHOLD_MS`, or on speaker change.
    the speaker-change condition is a documented no-op in phase 1: `Atom.
    speaker` is always `None` (no diarization yet, see `clipper/stages/
    atomize.py`'s module docstring), so two adjacent atoms can never have
    different non-`None` speakers -- same pattern atomize.py itself uses for
    its own speaker-change split condition.
    """
    if not atoms:
        return []

    regions: list[list[Atom]] = [[atoms[0]]]
    for atom in atoms[1:]:
        if atom.pause_before_ms >= _REGION_PAUSE_THRESHOLD_MS:
            regions.append([])
        regions[-1].append(atom)
    return regions


def _contiguous_runs_within_duration_bounds(region: list[Atom], cfg: Config) -> list[tuple[int, int]]:
    """every `(start_index, end_index)` pair of *region-local* indices whose
    atom span (`region[end_index].end_ms - region[start_index].start_ms`)
    fits `[cfg.min_duration_ms, cfg.max_duration_ms]`.

    runs may start and end anywhere within the region, not only at the
    region's own boundary: the spec's "every contiguous atom run inside a
    region" wording doesn't restrict where a run starts, and restricting to
    region-prefix runs would silently produce zero candidates for a region
    whose only duration-fitting content starts partway through it (e.g. a
    long region with a short valid clip buried in its middle). see
    DECISIONS.md.
    """
    runs: list[tuple[int, int]] = []
    region_len = len(region)
    for start_index in range(region_len):
        start_ms = region[start_index].start_ms
        for end_index in range(start_index, region_len):
            span_ms = region[end_index].end_ms - start_ms
            if span_ms > cfg.max_duration_ms:
                break  # span only grows as end_index increases; no point continuing
            if span_ms >= cfg.min_duration_ms:
                runs.append((start_index, end_index))
    return runs


def _select_representative_runs(
    runs: list[tuple[int, int]], region: list[Atom]
) -> list[tuple[int, int]]:
    """cap `runs` at `_MAX_CANDIDATES_PER_REGION`: the spec says "longest,
    shortest-valid, and two intermediate". implemented as: sort all valid
    runs by duration ascending, then take the shortest, the longest, and two
    runs at roughly the 1/3 and 2/3 points of that sorted order as a
    reasonable stand-in for "two intermediate" (the spec doesn't say exactly
    which two). duplicates (possible when the sorted list is short enough
    that the 1/3, 2/3, and endpoint picks land on the same run) are removed,
    order-preserving. no-op if there are already `<= 4` runs.
    """
    if len(runs) <= _MAX_CANDIDATES_PER_REGION:
        return runs

    by_duration = sorted(runs, key=lambda run: region[run[1]].end_ms - region[run[0]].start_ms)
    shortest = by_duration[0]
    longest = by_duration[-1]
    mid_a = by_duration[len(by_duration) // 3]
    mid_b = by_duration[(2 * len(by_duration)) // 3]

    selected: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for run in (shortest, mid_a, mid_b, longest):
        if run not in seen:
            seen.add(run)
            selected.append(run)
    return selected


def generate_discourse(atoms: list[Atom], cfg: Config) -> list[Candidate]:
    """g2: see module docstring."""
    candidates: list[Candidate] = []
    for region in _split_into_regions(atoms):
        runs = _contiguous_runs_within_duration_bounds(region, cfg)
        for start_index, end_index in _select_representative_runs(runs, region):
            candidates.append(
                build_candidate(region[start_index], region[end_index], "discourse")
            )
    return candidates
