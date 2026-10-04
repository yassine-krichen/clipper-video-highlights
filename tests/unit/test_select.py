"""tests for `clipper/stages/select.py` (spec section 4.11)."""

from __future__ import annotations

import json
from pathlib import Path

from clipper.config import Config
from clipper.stages.select import run_select, select_clips, timeline_proximity
from tests.unit.builders import make_atom, make_refined


class TestTimelineProximity:
    def test_within_120s_is_one(self) -> None:
        near = make_refined("n", 150_000, 180_000, 1)
        assert timeline_proximity(near, [make_refined("p", 0, 30_000, 1)]) == 1.0

    def test_beyond_ten_minutes_is_zero(self) -> None:
        far = make_refined("f", 700_000, 730_000, 1)
        assert timeline_proximity(far, [make_refined("p", 0, 30_000, 1)]) == 0.0

    def test_decays_linearly_between(self) -> None:
        # gap of 360s is halfway between 120s and 600s
        mid = make_refined("m", 390_000, 420_000, 1)
        assert timeline_proximity(mid, [make_refined("p", 0, 30_000, 1)]) == 0.5

    def test_no_picks_means_no_proximity(self) -> None:
        assert timeline_proximity(make_refined("m", 0, 1, 1), []) == 0.0


class TestSelectClips:
    def test_picks_best_scores_when_spread_and_distinct(self) -> None:
        cands = [
            make_refined("a", 0, 30_000, 90.0, embedding=[1.0, 0.0, 0.0]),
            make_refined("b", 1_000_000, 1_030_000, 80.0, embedding=[0.0, 1.0, 0.0]),
            make_refined("c", 2_000_000, 2_030_000, 70.0, embedding=[0.0, 0.0, 1.0]),
        ]
        assert [c.id for c in select_clips(cands, 2)] == ["a", "b"]

    def test_proximity_term_skips_a_clip_next_to_a_better_pick(self) -> None:
        cands = [
            make_refined("a", 0, 30_000, 90.0, embedding=[1.0, 0.0]),
            make_refined("near", 60_000, 90_000, 85.0, embedding=[0.0, 1.0]),
            make_refined("far", 2_000_000, 2_030_000, 80.0, embedding=[0.0, 1.0]),
        ]
        # near: 85 - 10 = 75 < far: 80
        assert [c.id for c in select_clips(cands, 2)] == ["a", "far"]

    def test_returns_fewer_when_not_enough_candidates(self) -> None:
        assert len(select_clips([make_refined("a", 0, 30_000, 90.0)], 5)) == 1


def test_run_select_writes_a_valid_manifest_and_markdown(tmp_path: Path) -> None:
    atoms = [make_atom(0, 0, 30_000, "the hiring mistake")]
    (tmp_path / "atoms.json").write_text(json.dumps([a.model_dump() for a in atoms]))
    (tmp_path / "input.json").write_text(
        json.dumps({"source_path": "C:/x/episode.mp4", "duration_ms": 120_000, "sha256": "abc"})
    )
    deduped = [make_refined("a", 0, 30_000, 80.0, embedding=[1.0, 0.0])]
    (tmp_path / "deduped.json").write_text(json.dumps([c.model_dump() for c in deduped]))

    run_select(tmp_path, Config(clip_count=3))

    manifest = json.loads((tmp_path / "selected.json").read_text())
    assert manifest["source"] == {"path": "episode.mp4", "duration_ms": 120_000, "sha256": "abc"}
    assert manifest["clips"][0]["id"] == "clip_01"
    assert manifest["clips"][0]["transcript"] == "the hiring mistake"
    assert manifest["clips"][0]["segments"] == [{"start_ms": 0, "end_ms": 30_000}]
    assert "clip_01: a title" in (tmp_path / "selected.md").read_text(encoding="utf-8")
