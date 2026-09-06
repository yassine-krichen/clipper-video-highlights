"""generate candidates stage (spec section 4.5): `run_dir/atoms.json` ->
`run_dir/candidates.json`.

runs all four phase-1 generators (g1 `llm_pass`, g2 `discourse`, g3
`lexical`, g4 `coverage`), merges their output, normalizes it, and writes
`candidates.json` plus per-generator counts to `stats.json`. g5-g7
(`query`/`audio_event`/`native_video`) are out of scope for phase 1 (spec
section 4.5/12).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from clipper.config import Config
from clipper.generators.coverage import generate_coverage
from clipper.generators.discourse import generate_discourse
from clipper.generators.lexical import generate_lexical
from clipper.generators.llm_pass import generate_llm_pass
from clipper.types import Atom, Candidate
from clipper.util.log import append_event, record_stat

_STAGE = "generate"
_ATOMS_JSON_FILENAME = "atoms.json"
_CANDIDATES_JSON_FILENAME = "candidates.json"


def _snap_to_atom_boundaries(candidate: Candidate, atoms_by_id: dict[int, Atom]) -> Candidate | None:
    """defensive normalization, not a geometric "snap to nearest boundary"
    operation: every generator already builds candidates directly from atom
    objects (`clipper/generators/base.py`'s `build_candidate`), so
    `atom_start`/`atom_end`/`start_ms`/`end_ms` are atom-aligned by
    construction and there is nothing to actually move. this instead
    re-derives `start_ms`/`end_ms` from the atoms `atom_start`/`atom_end`
    reference and validates the range, so a bug in one generator (a stale
    `start_ms`/`end_ms`, or an out-of-range atom id) is caught here instead
    of silently reaching `candidates.json`. returns `None` (dropped) if
    `atom_start`/`atom_end` doesn't resolve to a real atom, or
    `atom_start > atom_end`. see DECISIONS.md.
    """
    start_atom = atoms_by_id.get(candidate.atom_start)
    end_atom = atoms_by_id.get(candidate.atom_end)
    if start_atom is None or end_atom is None or candidate.atom_start > candidate.atom_end:
        return None
    return candidate.model_copy(update={"start_ms": start_atom.start_ms, "end_ms": end_atom.end_ms})


def _within_duration_bounds(candidate: Candidate, cfg: Config) -> bool:
    duration_ms = candidate.end_ms - candidate.start_ms
    return cfg.min_duration_ms <= duration_ms <= cfg.max_duration_ms


def _validate_and_filter(
    candidates: list[Candidate], atoms_by_id: dict[int, Atom], cfg: Config, run_dir: Path
) -> list[Candidate]:
    """run `_snap_to_atom_boundaries` then `_within_duration_bounds` over
    `candidates`, dropping (and logging) anything that fails either check.
    shared by both the pre-coverage (g1-g3) list and coverage's (g4) own
    output, so both go through identical validation before anything treats
    them as "final" -- see `run_generate`'s module-level ordering note on
    why this must run BEFORE `generate_coverage` sees the g1-g3 output."""
    filtered: list[Candidate] = []
    for candidate in candidates:
        snapped = _snap_to_atom_boundaries(candidate, atoms_by_id)
        if snapped is None:
            append_event(
                run_dir,
                _STAGE,
                "candidate_dropped_invalid_range",
                {
                    "generators": candidate.generators,
                    "atom_start": candidate.atom_start,
                    "atom_end": candidate.atom_end,
                    "reason": "unresolvable atom id or inverted range",
                },
            )
            continue
        if not _within_duration_bounds(snapped, cfg):
            continue
        filtered.append(snapped)
    return filtered


def _merge_identical_ranges(candidates: list[Candidate]) -> list[Candidate]:
    """merge candidates whose atom ranges are identical, unioning
    `generators` (spec section 4.5's normalization step). the first
    candidate seen for a given `(atom_start, atom_end)` is kept as the base
    (its `id`/`generator_hint`/`pre_score` survive into the merged result);
    only `generators` is combined, order-preserving and de-duplicated across
    every candidate sharing that range. this "first-seen-wins" behavior is
    incidental to the list concatenation order callers happen to pass in
    (llm_pass, then discourse, then lexical, then coverage) -- it is not a
    deliberate generator-priority policy. see DECISIONS.md."""
    merged: dict[tuple[int, int], Candidate] = {}
    for candidate in candidates:
        key = (candidate.atom_start, candidate.atom_end)
        existing = merged.get(key)
        if existing is None:
            merged[key] = candidate
            continue

        combined_generators = list(existing.generators)
        for generator in candidate.generators:
            if generator not in combined_generators:
                combined_generators.append(generator)
        merged[key] = existing.model_copy(update={"generators": combined_generators})

    return list(merged.values())


def run_generate(run_dir: Path, cfg: Config) -> None:
    """generate, merge, and normalize candidates (spec section 4.5),
    writing `run_dir/candidates.json` and per-generator counts to
    `stats.json`."""
    append_event(run_dir, _STAGE, "started", {})

    atoms = [
        Atom.model_validate(raw)
        for raw in json.loads((run_dir / _ATOMS_JSON_FILENAME).read_text(encoding="utf-8"))
    ]
    atoms_by_id = {atom.id: atom for atom in atoms}

    wall_start = time.monotonic()

    # g4 (coverage) runs last and needs g1-g3's combined output (spec
    # section 4.5: "after g1-g3, find any atom not inside any candidate").
    #
    # that combined output is validated/duration-filtered BEFORE coverage
    # sees it, not after: llm_pass's raw proposals have no duration-bound
    # enforcement at all, and lexical's variants only enforce one bound
    # each, so some raw g1-g3 candidates are destined to be dropped by
    # `_within_duration_bounds` later. if coverage ran against the raw,
    # pre-filter output (as this used to), an atom range could be
    # considered "covered" by a candidate that then gets filtered out,
    # leaving that range with zero surviving candidates in the final
    # `candidates.json` -- exactly the gap g4 exists to close. running
    # `_validate_and_filter` first means only candidates that will actually
    # survive into `candidates.json` count as "coverage". see DECISIONS.md.
    llm_pass_candidates = asyncio.run(generate_llm_pass(atoms, cfg, run_dir))
    discourse_candidates = generate_discourse(atoms, cfg)
    lexical_candidates = generate_lexical(atoms, cfg)
    pre_coverage_candidates = llm_pass_candidates + discourse_candidates + lexical_candidates
    filtered_pre_coverage = _validate_and_filter(pre_coverage_candidates, atoms_by_id, cfg, run_dir)

    coverage_candidates = generate_coverage(atoms, cfg, filtered_pre_coverage)
    filtered_coverage = _validate_and_filter(coverage_candidates, atoms_by_id, cfg, run_dir)

    per_generator_counts = {
        "llm_pass": len(llm_pass_candidates),
        "discourse": len(discourse_candidates),
        "lexical": len(lexical_candidates),
        "coverage": len(coverage_candidates),
    }

    merged_candidates = _merge_identical_ranges(filtered_pre_coverage + filtered_coverage)

    wall_ms = round((time.monotonic() - wall_start) * 1000)

    (run_dir / _CANDIDATES_JSON_FILENAME).write_text(
        json.dumps([c.model_dump() for c in merged_candidates], indent=2)
    )

    append_event(
        run_dir,
        _STAGE,
        "generated",
        {**per_generator_counts, "merged": len(merged_candidates)},
    )

    for generator_name, count in per_generator_counts.items():
        record_stat(run_dir, "candidates", generator_name, count)
    record_stat(run_dir, "candidates", "merged", len(merged_candidates))

    record_stat(run_dir, "stages", _STAGE, {"wall_ms": wall_ms})
