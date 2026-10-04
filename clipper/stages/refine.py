"""refine boundaries stage (spec section 4.8, contiguous only):
`judged.json` + `atoms.json` -> `refined.json`.

deterministic. applies the judge's suggested atom range, adds a short lead-in
and tail inside the silence around the range (so a boundary never lands inside
a word), fits the result to the duration bounds one atom at a time, computes
the boundary and silence penalties, and recomputes `score` (the judge-time
score assumed zero penalties, see `clipper/stages/judge.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

from clipper.config import Config
from clipper.stages.judge import compute_keyword_bonus, compute_score
from clipper.types import Atom, JudgedCandidate, RefinedCandidate, Segment
from clipper.util.log import append_event, record_stat
from clipper.util.silence import compute_timing_silence_ratio

_STAGE = "refine"
_ATOMS_JSON_FILENAME = "atoms.json"
_JUDGED_JSON_FILENAME = "judged.json"
_REFINED_JSON_FILENAME = "refined.json"

_MAX_LEAD_IN_MS = 300
_MAX_TAIL_MS = 400
# spec section 4.7: the judge may suggest at most 2 atoms outside the candidate.
_MAX_SUGGESTION_OVERSHOOT_ATOMS = 2
_BOUNDARY_PENALTY_PER_EDGE = 5.0
_SILENCE_PENALTY_MAX = 5.0

# openers that signal an atom leans on what came before it (spec section 4.8).
# a bare "that's" is deliberately absent: "that's a great story" stands alone,
# so only the "that's why" phrase counts.
_DEPENDENT_OPENER_WORDS = frozenset({"so", "because", "and", "but", "or", "then"})
_DEPENDENT_OPENER_PHRASES = ("that's why", "thats why")


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def starts_with_dependent_word(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered.startswith(_DEPENDENT_OPENER_PHRASES):
        return True
    first_word = lowered.split(" ", 1)[0].strip(",.!?;:")
    return first_word in _DEPENDENT_OPENER_WORDS


def _span_ms(start_atom: Atom, end_atom: Atom, duration_ms: int) -> tuple[int, int]:
    """atom range -> padded (start_ms, end_ms). padding is at most half the
    adjacent silence, so it can never reach the neighbouring atom's words."""
    start_ms = start_atom.start_ms - min(_MAX_LEAD_IN_MS, start_atom.pause_before_ms // 2)
    end_ms = end_atom.end_ms + min(_MAX_TAIL_MS, end_atom.pause_after_ms // 2)
    return max(0, start_ms), min(duration_ms, end_ms)


def _fit_to_duration_bounds(
    start_id: int,
    end_id: int,
    atoms_by_id: dict[int, Atom],
    max_atom_id: int,
    duration_ms: int,
    cfg: Config,
    log: list[str],
) -> tuple[int, int]:
    """shrink or grow the range one atom at a time until it fits (spec: prefer
    dropping a trailing atom over a leading one, prefer adding a following atom
    over a preceding one). stops if no move is possible; the gate rejects what
    still does not fit."""
    # a grow step can overshoot max and the following shrink step undo it,
    # so cap iterations instead of trusting convergence.
    for _ in range(len(atoms_by_id) * 2 + 2):
        start_ms, end_ms = _span_ms(atoms_by_id[start_id], atoms_by_id[end_id], duration_ms)
        length_ms = end_ms - start_ms

        if length_ms > cfg.max_duration_ms:
            if end_id == start_id:
                log.append(f"too long ({length_ms}ms): single atom, cannot shrink")
                return start_id, end_id
            log.append(f"too long ({length_ms}ms): dropped trailing atom {end_id}")
            end_id -= 1
        elif length_ms < cfg.min_duration_ms:
            if end_id < max_atom_id:
                end_id += 1
                log.append(f"too short ({length_ms}ms): added following atom {end_id}")
            elif start_id > 0:
                start_id -= 1
                log.append(f"too short ({length_ms}ms): added preceding atom {start_id}")
            else:
                log.append(f"too short ({length_ms}ms): no atoms left to add")
                return start_id, end_id
        else:
            return start_id, end_id

    log.append("duration fitting did not converge; left as is")
    return start_id, end_id


def refine_candidate(
    judged: JudgedCandidate,
    atoms_by_id: dict[int, Atom],
    max_atom_id: int,
    duration_ms: int,
    cfg: Config,
) -> RefinedCandidate:
    log: list[str] = []

    # the model is told to stay within 2 atoms of the candidate but is not trusted to.
    low = max(0, judged.atom_start - _MAX_SUGGESTION_OVERSHOOT_ATOMS)
    high = min(max_atom_id, judged.atom_end + _MAX_SUGGESTION_OVERSHOOT_ATOMS)
    start_id = _clamp(judged.judge.suggested_atom_start, low, high)
    end_id = _clamp(judged.judge.suggested_atom_end, low, high)
    if start_id > end_id:
        log.append("judge suggestion inverted: fell back to the candidate's own range")
        start_id, end_id = judged.atom_start, judged.atom_end
    elif (start_id, end_id) != (judged.atom_start, judged.atom_end):
        log.append(f"applied judge suggestion: atoms {start_id}..{end_id}")

    start_id, end_id = _fit_to_duration_bounds(
        start_id, end_id, atoms_by_id, max_atom_id, duration_ms, cfg, log
    )
    start_ms, end_ms = _span_ms(atoms_by_id[start_id], atoms_by_id[end_id], duration_ms)

    boundary_penalty = _BOUNDARY_PENALTY_PER_EDGE * (
        starts_with_dependent_word(atoms_by_id[start_id].text)
        + starts_with_dependent_word(atoms_by_id[end_id].text)
    )
    silence_ratio = compute_timing_silence_ratio(start_id, end_id, atoms_by_id)
    penalties = {
        "boundary": boundary_penalty,
        "silence": _SILENCE_PENALTY_MAX * silence_ratio,
        "redundancy": 0.0,
    }

    text = " ".join(atoms_by_id[i].text for i in range(start_id, end_id + 1))
    score = compute_score(
        judged.judge,
        cfg.resolved_scoring_weights(),
        compute_keyword_bonus(text, cfg.keywords),
        penalties,
    )

    fields = judged.model_dump()
    fields.update(
        atom_start=start_id,
        atom_end=end_id,
        start_ms=start_ms,
        end_ms=end_ms,
        score=score,
    )
    return RefinedCandidate(
        **fields,
        segments=[Segment(start_ms=start_ms, end_ms=end_ms)],
        refinement_log=log,
        penalties=penalties,
        embedding=None,
    )


def run_refine(run_dir: Path, cfg: Config) -> None:
    """refine every candidate in `run_dir/judged.json`, writing
    `run_dir/refined.json`."""
    append_event(run_dir, _STAGE, "started", {})

    atoms = [
        Atom.model_validate(raw)
        for raw in json.loads((run_dir / _ATOMS_JSON_FILENAME).read_text(encoding="utf-8"))
    ]
    atoms_by_id = {atom.id: atom for atom in atoms}
    max_atom_id = max(atoms_by_id) if atoms_by_id else -1
    duration_ms = atoms[-1].end_ms if atoms else 0

    judged = [
        JudgedCandidate.model_validate(raw)
        for raw in json.loads((run_dir / _JUDGED_JSON_FILENAME).read_text(encoding="utf-8"))
    ]
    refined = [refine_candidate(c, atoms_by_id, max_atom_id, duration_ms, cfg) for c in judged]

    (run_dir / _REFINED_JSON_FILENAME).write_text(
        json.dumps([c.model_dump() for c in refined], indent=2)
    )
    append_event(run_dir, _STAGE, "refined", {"count": len(refined)})
    record_stat(run_dir, "candidates", "refined", len(refined))
