"""prefilter stage (spec section 4.6): `run_dir/candidates.json` ->
(pre_score computed, top `judge_cap` kept, every `llm_pass` candidate always
kept) -> `run_dir/candidates.json`, overwritten in place.

deterministic and cheap by design (spec: "this stage exists to bound cost,
not to be clever") -- no llm calls, no audio i/o.
"""

from __future__ import annotations

import json
from pathlib import Path

from clipper.config import Config
from clipper.types import Atom, Candidate
from clipper.util.log import append_event, record_stat
from clipper.util.silence import compute_timing_silence_ratio

_STAGE = "prefilter"
_ATOMS_JSON_FILENAME = "atoms.json"
_CANDIDATES_JSON_FILENAME = "candidates.json"


def _duration_fit(duration_ms: int, cfg: Config) -> float:
    """triangular/tent function over `[min_duration_ms, max_duration_ms]`:
    `1.0` at the midpoint, `0.0` at either edge, linear in between (spec:
    "1 at the middle of the range, 0 at the edges"). clamped to `[0, 1]`
    defensively for a duration outside the configured bounds (shouldn't
    happen -- `generate.py` already filters those out -- but cheap to guard)."""
    mid = (cfg.min_duration_ms + cfg.max_duration_ms) / 2
    half_range = (cfg.max_duration_ms - cfg.min_duration_ms) / 2
    if half_range <= 0:
        return 0.0
    fit = 1 - abs(duration_ms - mid) / half_range
    return max(0.0, min(1.0, fit))


def _lexical_hit(candidate: Candidate) -> float:
    """`1.0` if this candidate was (among others) proposed BY the lexical
    hook-matching generator, else `0.0`. reuses `generators` membership
    already present on the candidate rather than re-scanning the spanned
    atoms' text for hook patterns independently -- see DECISIONS.md."""
    return 1.0 if "lexical" in candidate.generators else 0.0


def _compute_pre_score(candidate: Candidate, atoms_by_id: dict[int, Atom], cfg: Config) -> float:
    """spec section 4.6's exact formula. clamped to `[0, 1]` defensively --
    each term is individually bounded to `[0, 1]` and the weights sum to
    1.0, so this should be a no-op in practice."""
    llm_pass_term = 0.40 * (1 if "llm_pass" in candidate.generators else 0)
    duration_term = 0.20 * _duration_fit(candidate.end_ms - candidate.start_ms, cfg)
    n_generators_term = 0.15 * min(1, len(candidate.generators) / 3)
    silence_ratio = compute_timing_silence_ratio(candidate.atom_start, candidate.atom_end, atoms_by_id)
    silence_term = 0.15 * (1 - silence_ratio)
    lexical_term = 0.10 * _lexical_hit(candidate)

    score = llm_pass_term + duration_term + n_generators_term + silence_term + lexical_term
    return max(0.0, min(1.0, score))


def _select_top_and_llm_pass(candidates: list[Candidate], judge_cap: int) -> list[Candidate]:
    """sort by `pre_score` descending, keep the top `judge_cap`, then add
    back any `llm_pass` candidate that fell outside that cutoff (spec:
    "always keep every llm_pass candidate even if that exceeds the cap").
    order of the returned list is score-descending; re-added llm_pass
    stragglers are appended at the end, kept in their own score-descending
    order."""
    ranked = sorted(candidates, key=lambda c: c.pre_score, reverse=True)
    kept = ranked[:judge_cap]
    kept_ids = {c.id for c in kept}

    dropped_llm_pass = [
        c for c in ranked[judge_cap:] if "llm_pass" in c.generators and c.id not in kept_ids
    ]
    return kept + dropped_llm_pass


def run_prefilter(run_dir: Path, cfg: Config) -> None:
    """compute `pre_score` for every candidate in `run_dir/candidates.json`,
    keep the top `cfg.judge_cap` plus every `llm_pass` candidate regardless
    of rank, and overwrite `run_dir/candidates.json` with the result (spec
    section 2.2 names no separate intermediate file for this stage -- see
    DECISIONS.md)."""
    append_event(run_dir, _STAGE, "started", {})

    atoms = [
        Atom.model_validate(raw)
        for raw in json.loads((run_dir / _ATOMS_JSON_FILENAME).read_text(encoding="utf-8"))
    ]
    atoms_by_id = {atom.id: atom for atom in atoms}

    candidates = [
        Candidate.model_validate(raw)
        for raw in json.loads((run_dir / _CANDIDATES_JSON_FILENAME).read_text(encoding="utf-8"))
    ]

    scored = [
        candidate.model_copy(update={"pre_score": _compute_pre_score(candidate, atoms_by_id, cfg)})
        for candidate in candidates
    ]

    kept = _select_top_and_llm_pass(scored, cfg.judge_cap)

    before_count = len(scored)
    after_count = len(kept)
    cap_exceeded_by_llm_pass = after_count > cfg.judge_cap

    (run_dir / _CANDIDATES_JSON_FILENAME).write_text(
        json.dumps([c.model_dump() for c in kept], indent=2)
    )

    append_event(
        run_dir,
        _STAGE,
        "prefiltered",
        {
            "before": before_count,
            "after": after_count,
            "dropped": before_count - after_count,
            "judge_cap": cfg.judge_cap,
            "cap_exceeded_by_llm_pass": cap_exceeded_by_llm_pass,
        },
    )

    record_stat(run_dir, "candidates", "after_prefilter", after_count)
