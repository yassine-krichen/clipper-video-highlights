"""tests for `clipper/stages/gate.py` (spec section 4.9)."""

from __future__ import annotations

import json
from pathlib import Path

from clipper.config import Config
from clipper.stages.gate import gate_reason, run_gate
from tests.unit.builders import make_atom, make_refined

_CFG = Config(min_duration_ms=10_000, max_duration_ms=60_000)
_ATOMS = {0: make_atom(0, 0, 30_000, "a perfectly fine long sentence " * 4)}


class TestGateReason:
    def test_passes_a_good_clip(self) -> None:
        assert gate_reason(make_refined("c", 0, 30_000, 70.0), _ATOMS, _CFG) is None

    def test_rejects_standalone_level_3(self) -> None:
        reason = gate_reason(make_refined("c", 0, 30_000, 90.0, level=3), _ATOMS, _CFG)
        assert reason is not None and "standalone_level 3" in reason

    def test_rejects_duration_outside_bounds(self) -> None:
        reason = gate_reason(make_refined("c", 0, 5_000, 90.0), _ATOMS, _CFG)
        assert reason is not None and "duration" in reason

    def test_rejects_score_below_forty_but_keeps_exactly_forty(self) -> None:
        assert gate_reason(make_refined("c", 0, 30_000, 39.9), _ATOMS, _CFG) is not None
        assert gate_reason(make_refined("c", 0, 30_000, 40.0), _ATOMS, _CFG) is None

    def test_rejects_short_low_confidence_clip_only(self) -> None:
        short_bad = {0: make_atom(0, 0, 30_000, "um yeah ok", confidence=0.2)}
        long_bad = {0: make_atom(0, 0, 30_000, "word " * 20, confidence=0.2)}
        candidate = make_refined("c", 0, 30_000, 80.0)

        assert gate_reason(candidate, short_bad, _CFG) is not None
        assert gate_reason(candidate, long_bad, _CFG) is None


def test_run_gate_writes_survivors_and_a_reason_for_every_reject(tmp_path: Path) -> None:
    (tmp_path / "atoms.json").write_text(json.dumps([_ATOMS[0].model_dump()]))
    refined = [make_refined("good", 0, 30_000, 80.0), make_refined("bad", 0, 30_000, 10.0)]
    (tmp_path / "refined.json").write_text(json.dumps([c.model_dump() for c in refined]))

    run_gate(tmp_path, _CFG)

    gated = json.loads((tmp_path / "gated.json").read_text())
    rejected = json.loads((tmp_path / "rejected.json").read_text())
    assert [c["id"] for c in gated] == ["good"]
    assert rejected[0]["candidate_id"] == "bad" and rejected[0]["stage"] == "gate"
