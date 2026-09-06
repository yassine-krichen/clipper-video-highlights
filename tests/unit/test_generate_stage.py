"""tests for `clipper/stages/generate.py` (spec section 4.5).

focuses on the STAGE's own orchestration and normalization logic (duration-
bounds filtering, identical-range merging with `generators` union, snap/
validate defensiveness), not re-testing each generator -- those have their
own test files. the last section stubs all four generator functions at the
module level and exercises `run_generate`'s real file i/o.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipper.config import Config
from clipper.stages import generate as generate_stage
from clipper.types import Atom, Candidate, Word
from clipper.util.log import read_events, read_stats


def _atom(atom_id: int, start_ms: int, end_ms: int) -> Atom:
    return Atom(
        id=atom_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text=f"atom {atom_id}",
        words=[Word(text=f"atom{atom_id}", start_ms=start_ms, end_ms=end_ms)],
        pause_before_ms=0,
        pause_after_ms=0,
    )


def _candidate(
    generator: str,
    atom_start: int,
    atom_end: int,
    start_ms: int = 0,
    end_ms: int = 0,
    *,
    candidate_id: str | None = None,
) -> Candidate:
    return Candidate(
        id=candidate_id or f"{generator}_{atom_start}_{atom_end}",
        atom_start=atom_start,
        atom_end=atom_end,
        start_ms=start_ms,
        end_ms=end_ms,
        generators=[generator],
        generator_hint=None,
        pre_score=0.0,
    )


class TestSnapToAtomBoundaries:
    def test_re_derives_start_and_end_ms_from_referenced_atoms(self) -> None:
        atoms_by_id = {0: _atom(0, 0, 1000), 1: _atom(1, 1000, 5000)}
        candidate = _candidate("discourse", 0, 1, start_ms=999, end_ms=4999)  # deliberately stale

        snapped = generate_stage._snap_to_atom_boundaries(candidate, atoms_by_id)

        assert snapped is not None
        assert snapped.start_ms == 0
        assert snapped.end_ms == 5000

    def test_returns_none_for_an_atom_id_that_does_not_exist(self) -> None:
        atoms_by_id = {0: _atom(0, 0, 1000)}
        candidate = _candidate("discourse", 0, 5, start_ms=0, end_ms=5000)

        assert generate_stage._snap_to_atom_boundaries(candidate, atoms_by_id) is None

    def test_returns_none_for_an_inverted_range(self) -> None:
        atoms_by_id = {0: _atom(0, 0, 1000), 1: _atom(1, 1000, 5000)}
        candidate = _candidate("discourse", 1, 0, start_ms=1000, end_ms=1000)

        assert generate_stage._snap_to_atom_boundaries(candidate, atoms_by_id) is None


class TestDurationBoundsFiltering:
    def test_candidate_within_bounds_is_kept(self) -> None:
        cfg = Config(min_duration_ms=10_000, max_duration_ms=90_000)
        candidate = _candidate("discourse", 0, 1, start_ms=0, end_ms=30_000)

        assert generate_stage._within_duration_bounds(candidate, cfg) is True

    def test_candidate_shorter_than_min_duration_is_dropped(self) -> None:
        cfg = Config(min_duration_ms=10_000, max_duration_ms=90_000)
        candidate = _candidate("discourse", 0, 1, start_ms=0, end_ms=5_000)

        assert generate_stage._within_duration_bounds(candidate, cfg) is False

    def test_candidate_longer_than_max_duration_is_dropped(self) -> None:
        cfg = Config(min_duration_ms=10_000, max_duration_ms=90_000)
        candidate = _candidate("discourse", 0, 1, start_ms=0, end_ms=100_000)

        assert generate_stage._within_duration_bounds(candidate, cfg) is False


class TestMergeIdenticalRanges:
    def test_candidates_with_identical_atom_ranges_are_merged_with_generators_unioned(
        self,
    ) -> None:
        a = _candidate("llm_pass", 0, 3, candidate_id="llm_pass_0_3")
        b = _candidate("discourse", 0, 3, candidate_id="discourse_0_3")

        merged = generate_stage._merge_identical_ranges([a, b])

        assert len(merged) == 1
        assert merged[0].generators == ["llm_pass", "discourse"]
        assert merged[0].id == "llm_pass_0_3"  # first-seen candidate is kept as the base

    def test_candidates_with_different_atom_ranges_are_not_merged(self) -> None:
        a = _candidate("llm_pass", 0, 3)
        b = _candidate("discourse", 4, 7)

        merged = generate_stage._merge_identical_ranges([a, b])

        assert len(merged) == 2

    def test_merging_the_same_generator_twice_does_not_duplicate_it_in_the_list(self) -> None:
        a = _candidate("discourse", 0, 3, candidate_id="discourse_a")
        b = _candidate("discourse", 0, 3, candidate_id="discourse_b")

        merged = generate_stage._merge_identical_ranges([a, b])

        assert len(merged) == 1
        assert merged[0].generators == ["discourse"]


class TestRunGenerateOrchestration:
    def test_writes_candidates_json_and_stats_with_all_generators_stubbed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        atoms = [_atom(i, i * 10_000, i * 10_000 + 10_000) for i in range(4)]
        (run_dir / "atoms.json").write_text(json.dumps([a.model_dump() for a in atoms]))

        # min_duration_ms=15_000 so a single-atom (10_000ms) candidate is
        # below bounds and dropped, while a two-atom (20_000ms) one survives
        # -- note `_snap_to_atom_boundaries` re-derives `start_ms`/`end_ms`
        # from the atoms an id range actually covers, so a stub's own
        # start_ms/end_ms values don't matter for bounds filtering, only
        # atom_start/atom_end do.
        cfg = Config(min_duration_ms=15_000, max_duration_ms=50_000)

        async def _stub_generate_llm_pass(atoms, cfg, run_dir):
            return [_candidate("llm_pass", 0, 1, start_ms=0, end_ms=20_000)]

        def _stub_generate_discourse(atoms, cfg):
            return [
                _candidate("discourse", 0, 1, start_ms=0, end_ms=20_000),  # duplicate range of llm_pass
                _candidate("discourse", 2, 3, start_ms=20_000, end_ms=40_000),
            ]

        def _stub_generate_lexical(atoms, cfg):
            return []

        def _stub_generate_coverage(atoms, cfg, existing):
            return [_candidate("coverage", 3, 3, start_ms=30_000, end_ms=31_000)]  # single atom, too short, dropped

        monkeypatch.setattr(generate_stage, "generate_llm_pass", _stub_generate_llm_pass)
        monkeypatch.setattr(generate_stage, "generate_discourse", _stub_generate_discourse)
        monkeypatch.setattr(generate_stage, "generate_lexical", _stub_generate_lexical)
        monkeypatch.setattr(generate_stage, "generate_coverage", _stub_generate_coverage)

        generate_stage.run_generate(run_dir, cfg)

        candidates = json.loads((run_dir / "candidates.json").read_text())
        assert len(candidates) == 2  # merged (0,1) + (2,3); the too-short coverage one is dropped

        merged_zero_one = next(c for c in candidates if c["atom_start"] == 0)
        assert set(merged_zero_one["generators"]) == {"llm_pass", "discourse"}

        stats = read_stats(run_dir)
        assert stats["candidates"]["llm_pass"] == 1
        assert stats["candidates"]["discourse"] == 2
        assert stats["candidates"]["lexical"] == 0
        assert stats["candidates"]["coverage"] == 1
        assert stats["candidates"]["merged"] == 2
        assert "wall_ms" in stats["stages"]["generate"]

        events = read_events(run_dir, stage="generate")
        event_names = [e["event"] for e in events]
        assert "started" in event_names
        assert "generated" in event_names

    def test_coverage_generator_receives_the_combined_g1_g3_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        atoms = [_atom(0, 0, 10_000)]
        (run_dir / "atoms.json").write_text(json.dumps([a.model_dump() for a in atoms]))
        cfg = Config(min_duration_ms=1_000, max_duration_ms=50_000)

        captured: dict[str, object] = {}

        async def _stub_generate_llm_pass(atoms, cfg, run_dir):
            return [_candidate("llm_pass", 0, 0, start_ms=0, end_ms=10_000)]

        def _stub_generate_discourse(atoms, cfg):
            return []

        def _stub_generate_lexical(atoms, cfg):
            return []

        def _stub_generate_coverage(atoms, cfg, existing):
            captured["existing"] = existing
            return []

        monkeypatch.setattr(generate_stage, "generate_llm_pass", _stub_generate_llm_pass)
        monkeypatch.setattr(generate_stage, "generate_discourse", _stub_generate_discourse)
        monkeypatch.setattr(generate_stage, "generate_lexical", _stub_generate_lexical)
        monkeypatch.setattr(generate_stage, "generate_coverage", _stub_generate_coverage)

        generate_stage.run_generate(run_dir, cfg)

        existing = captured["existing"]
        assert len(existing) == 1
        assert existing[0].generators == ["llm_pass"]


class TestCoverageRunsAfterDurationBoundsFiltering:
    """blocker 2 fix: coverage must run against g1-g3 output that has
    ALREADY passed the duration-bounds filter, not the raw output -- an
    out-of-bounds candidate is destined to be dropped from
    `candidates.json` anyway, so it must not be allowed to "count" as
    covering an atom range and starve that range of any surviving
    candidate. uses the real `generate_coverage` (g4), not a stub, since
    the point being proven is about the real g4 algorithm's interaction
    with filtering order.
    """

    def test_an_atom_range_only_proposed_out_of_bounds_still_ends_up_with_a_candidate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # three atoms: llm_pass "covers" atoms 0-1 with a 4_000ms candidate
        # -- below `min_duration_ms` (5_000ms), so it's destined to be
        # filtered out. atom 2 alone (4_000-9_000ms = 5_000ms) is never
        # proposed by anyone, but happens to already meet min_duration_ms
        # on its own.
        atoms = [
            _atom(0, 0, 2_000),
            _atom(1, 2_000, 4_000),
            _atom(2, 4_000, 9_000),
        ]
        (run_dir / "atoms.json").write_text(json.dumps([a.model_dump() for a in atoms]))
        cfg = Config(min_duration_ms=5_000, max_duration_ms=50_000)

        async def _stub_generate_llm_pass(atoms, cfg, run_dir):
            return [_candidate("llm_pass", 0, 1, start_ms=0, end_ms=4_000)]  # too short, will be dropped

        def _stub_generate_discourse(atoms, cfg):
            return []

        def _stub_generate_lexical(atoms, cfg):
            return []

        monkeypatch.setattr(generate_stage, "generate_llm_pass", _stub_generate_llm_pass)
        monkeypatch.setattr(generate_stage, "generate_discourse", _stub_generate_discourse)
        monkeypatch.setattr(generate_stage, "generate_lexical", _stub_generate_lexical)
        # `generate_coverage` (g4) itself is NOT stubbed here -- the real
        # algorithm is what proves the fix.

        generate_stage.run_generate(run_dir, cfg)

        candidates = json.loads((run_dir / "candidates.json").read_text())

        # under the OLD (blocked) pipeline order, coverage would have seen
        # the raw, pre-filter llm_pass candidate as already "covering" atoms
        # 0 and 1, generated nothing for them, and then that llm_pass
        # candidate itself would have been dropped by the duration-bounds
        # filter -- leaving atoms 0 and 1 with ZERO surviving candidates.
        #
        # with the fix, the too-short llm_pass candidate is filtered out
        # BEFORE coverage runs, so atoms 0 and 1 register as uncovered and
        # coverage emits a real (bounds-compliant) candidate that includes
        # them.
        assert len(candidates) == 1
        covering = candidates[0]
        assert covering["generators"] == ["coverage"]
        assert covering["atom_start"] <= 0
        assert covering["atom_end"] >= 1
