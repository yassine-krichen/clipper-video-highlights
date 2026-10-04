"""tests for `clipper/stages/dedupe.py` (spec section 4.10)."""

from __future__ import annotations

import json
from pathlib import Path

from clipper.config import Config
from clipper.stages.dedupe import dedupe_semantic, dedupe_temporal, interval_iou, run_dedupe
from tests.unit.builders import make_atom, make_refined


class TestIntervalIou:
    def test_disjoint_intervals_have_zero_iou(self) -> None:
        assert interval_iou(make_refined("a", 0, 10, 1), make_refined("b", 20, 30, 1)) == 0.0

    def test_half_overlap_is_one_third(self) -> None:
        assert interval_iou(make_refined("a", 0, 20, 1), make_refined("b", 10, 30, 1)) == 10 / 30


class TestDedupeTemporal:
    def test_keeps_higher_score_of_overlapping_pair(self) -> None:
        low = make_refined("low", 0, 30_000, 50.0)
        high = make_refined("high", 5_000, 35_000, 80.0)

        kept, rejected = dedupe_temporal([low, high])

        assert [c.id for c in kept] == ["high"]
        assert rejected[0]["candidate_id"] == "low" and "high" in rejected[0]["reason"]

    def test_keeps_both_when_iou_is_below_threshold(self) -> None:
        # overlap 10 of union 30 is iou 0.333, under 0.35
        a = make_refined("a", 0, 20_000, 60.0)
        b = make_refined("b", 10_000, 30_000, 70.0)

        kept, rejected = dedupe_temporal([a, b])

        assert len(kept) == 2 and rejected == []


class TestDedupeSemantic:
    def test_rejects_near_duplicate_and_keeps_distinct(self) -> None:
        atoms = {0: make_atom(0, 0, 30_000, "same story")}
        a = make_refined("a", 0, 30_000, 80.0, embedding=[1.0, 0.0])
        b = make_refined("b", 100_000, 130_000, 70.0, embedding=[0.99, 0.141])
        c = make_refined("c", 200_000, 230_000, 60.0, embedding=[0.0, 1.0])

        kept, rejected = dedupe_semantic([a, b, c], Config(), atoms)

        assert [x.id for x in kept] == ["a", "c"]
        assert rejected[0]["candidate_id"] == "b"
        assert "semantic duplicate of a" in rejected[0]["reason"]


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "story" in t else [0.0, 1.0] for t in texts]


def test_run_dedupe_end_to_end_with_fake_embedder(tmp_path: Path) -> None:
    atoms = [
        make_atom(0, 0, 30_000, "a story about hiring"),
        make_atom(1, 100_000, 130_000, "another story about hiring"),
        make_atom(2, 200_000, 230_000, "unrelated topic"),
    ]
    (tmp_path / "atoms.json").write_text(json.dumps([a.model_dump() for a in atoms]))
    gated = [
        make_refined("a", 0, 30_000, 80.0, atom_start=0, atom_end=0),
        make_refined("b", 100_000, 130_000, 70.0, atom_start=1, atom_end=1),
        make_refined("c", 200_000, 230_000, 60.0, atom_start=2, atom_end=2),
    ]
    (tmp_path / "gated.json").write_text(json.dumps([c.model_dump() for c in gated]))
    (tmp_path / "rejected.json").write_text(
        json.dumps([{"candidate_id": "z", "stage": "gate", "reason": "x"}])
    )

    run_dedupe(tmp_path, Config(), embedder=_FakeEmbedder())

    kept = json.loads((tmp_path / "deduped.json").read_text())
    rejected = json.loads((tmp_path / "rejected.json").read_text())
    assert [c["id"] for c in kept] == ["a", "c"]
    assert kept[0]["embedding"] == [1.0, 0.0]
    # gate's earlier reject survives, dedupe's is appended
    assert {r["candidate_id"] for r in rejected} == {"z", "b"}
