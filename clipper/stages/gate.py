"""gate stage (spec section 4.9): `refined.json` -> `gated.json`.

hard rejects only; every reject is logged and written to `rejected.json`
with its reason so `clipper inspect` can show each candidate's fate.
"""

from __future__ import annotations

import json
from pathlib import Path

from clipper.config import Config
from clipper.types import Atom, RefinedCandidate
from clipper.util.log import append_event, record_stat

_STAGE = "gate"
_ATOMS_JSON_FILENAME = "atoms.json"
_REFINED_JSON_FILENAME = "refined.json"
_GATED_JSON_FILENAME = "gated.json"
_REJECTED_JSON_FILENAME = "rejected.json"

_MIN_SCORE = 40.0
_MIN_MEAN_CONFIDENCE = 0.5
_LOW_CONFIDENCE_WORD_FLOOR = 15


def gate_reason(candidate: RefinedCandidate, atoms_by_id: dict[int, Atom], cfg: Config) -> str | None:
    """the first hard-reject reason that applies, or `None` if it passes."""
    if candidate.judge.standalone_level == 3:
        return "standalone_level 3: depends on prior context"

    duration_ms = sum(s.end_ms - s.start_ms for s in candidate.segments)
    if not cfg.min_duration_ms <= duration_ms <= cfg.max_duration_ms:
        return f"duration {duration_ms}ms outside [{cfg.min_duration_ms}, {cfg.max_duration_ms}]"

    if candidate.score < _MIN_SCORE:
        return f"score {candidate.score:.1f} below {_MIN_SCORE:.0f}"

    words = [
        word
        for i in range(candidate.atom_start, candidate.atom_end + 1)
        for word in atoms_by_id[i].words
    ]
    confidences = [w.confidence for w in words if w.confidence is not None]
    if (
        confidences
        and len(words) < _LOW_CONFIDENCE_WORD_FLOOR
        and sum(confidences) / len(confidences) < _MIN_MEAN_CONFIDENCE
    ):
        return "low transcript confidence on a very short clip"

    return None


def run_gate(run_dir: Path, cfg: Config) -> None:
    append_event(run_dir, _STAGE, "started", {})

    atoms_by_id = {
        raw["id"]: Atom.model_validate(raw)
        for raw in json.loads((run_dir / _ATOMS_JSON_FILENAME).read_text(encoding="utf-8"))
    }
    refined = [
        RefinedCandidate.model_validate(raw)
        for raw in json.loads((run_dir / _REFINED_JSON_FILENAME).read_text(encoding="utf-8"))
    ]

    kept: list[RefinedCandidate] = []
    rejected: list[dict[str, str]] = []
    for candidate in refined:
        reason = gate_reason(candidate, atoms_by_id, cfg)
        if reason is None:
            kept.append(candidate)
            continue
        rejected.append({"candidate_id": candidate.id, "stage": _STAGE, "reason": reason})
        append_event(run_dir, _STAGE, "rejected", {"candidate_id": candidate.id, "reason": reason})

    (run_dir / _GATED_JSON_FILENAME).write_text(json.dumps([c.model_dump() for c in kept], indent=2))
    (run_dir / _REJECTED_JSON_FILENAME).write_text(json.dumps(rejected, indent=2))
    record_stat(run_dir, "candidates", "after_gate", len(kept))
