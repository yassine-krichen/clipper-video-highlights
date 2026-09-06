"""tests for `clipper/stages/prefilter.py` (spec section 4.6).

fully deterministic, hand-built `Candidate`/`Atom` lists -- no llm, no
network, no audio i/o needed (this stage is deliberately "deterministic,
cheap", per the spec).
"""

from __future__ import annotations

import json
from pathlib import Path

from clipper.config import Config
from clipper.stages import prefilter as prefilter_stage
from clipper.types import Atom, Candidate, Word
from clipper.util.log import read_events, read_stats


def _atom(
    atom_id: int,
    start_ms: int,
    end_ms: int,
    *,
    pause_before_ms: int = 0,
    pause_after_ms: int = 0,
) -> Atom:
    return Atom(
        id=atom_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text=f"atom {atom_id}",
        words=[Word(text=f"atom{atom_id}", start_ms=start_ms, end_ms=end_ms)],
        pause_before_ms=pause_before_ms,
        pause_after_ms=pause_after_ms,
    )


def _candidate(
    generators: list[str],
    atom_start: int,
    atom_end: int,
    start_ms: int,
    end_ms: int,
    *,
    candidate_id: str | None = None,
) -> Candidate:
    return Candidate(
        id=candidate_id or f"{'-'.join(generators)}_{atom_start}_{atom_end}",
        atom_start=atom_start,
        atom_end=atom_end,
        start_ms=start_ms,
        end_ms=end_ms,
        generators=generators,
        generator_hint=None,
        pre_score=0.0,
    )


class TestDurationFit:
    def test_exact_midpoint_of_range_scores_one(self) -> None:
        cfg = Config(min_duration_ms=20_000, max_duration_ms=80_000)
        assert prefilter_stage._duration_fit(50_000, cfg) == 1.0

    def test_exact_min_edge_scores_zero(self) -> None:
        cfg = Config(min_duration_ms=20_000, max_duration_ms=80_000)
        assert prefilter_stage._duration_fit(20_000, cfg) == 0.0

    def test_exact_max_edge_scores_zero(self) -> None:
        cfg = Config(min_duration_ms=20_000, max_duration_ms=80_000)
        assert prefilter_stage._duration_fit(80_000, cfg) == 0.0

    def test_quarter_point_scores_half(self) -> None:
        cfg = Config(min_duration_ms=20_000, max_duration_ms=80_000)
        # 35_000 is halfway between the min edge (20_000) and the midpoint
        # (50_000) -- fit should be 0.5, linear in between.
        assert prefilter_stage._duration_fit(35_000, cfg) == 0.5


class TestLexicalHit:
    def test_candidate_proposed_by_lexical_generator_scores_one(self) -> None:
        candidate = _candidate(["lexical"], 0, 1, 0, 30_000)
        assert prefilter_stage._lexical_hit(candidate) == 1.0

    def test_candidate_not_proposed_by_lexical_generator_scores_zero(self) -> None:
        candidate = _candidate(["discourse"], 0, 1, 0, 30_000)
        assert prefilter_stage._lexical_hit(candidate) == 0.0


class TestComputePreScoreTermIsolation:
    """each sub-case isolates one term by zeroing out the others as much as
    the formula allows, so a broken term shows up as a specific, obvious
    assertion failure."""

    def test_llm_pass_term_only(self) -> None:
        # duration at an edge (duration_fit=0), single non-lexical generator
        # (n_generators term nonzero but that's unavoidable -- min 1
        # generator always contributes 0.15/3), no atoms span (silence_ratio
        # computed against a single atom with no internal pause -> 0 pause).
        cfg = Config(min_duration_ms=20_000, max_duration_ms=80_000)
        atoms_by_id = {0: _atom(0, 0, 20_000)}
        candidate = _candidate(["llm_pass"], 0, 0, 0, 20_000)

        score = prefilter_stage._compute_pre_score(candidate, atoms_by_id, cfg)

        expected = 0.40 * 1 + 0.20 * 0 + 0.15 * (1 / 3) + 0.15 * 1 + 0.10 * 0
        assert score == expected

    def test_duration_fit_term_at_midpoint_with_no_llm_pass_no_lexical(self) -> None:
        cfg = Config(min_duration_ms=20_000, max_duration_ms=80_000)
        atoms_by_id = {0: _atom(0, 0, 50_000)}
        candidate = _candidate(["discourse"], 0, 0, 0, 50_000)

        score = prefilter_stage._compute_pre_score(candidate, atoms_by_id, cfg)

        expected = 0.40 * 0 + 0.20 * 1 + 0.15 * (1 / 3) + 0.15 * 1 + 0.10 * 0
        assert score == expected


class TestNGeneratorsScaling:
    def _score_with_n_generators(self, generators: list[str]) -> float:
        cfg = Config(min_duration_ms=20_000, max_duration_ms=80_000)
        atoms_by_id = {0: _atom(0, 0, 20_000)}
        candidate = _candidate(generators, 0, 0, 0, 20_000)
        return prefilter_stage._compute_pre_score(candidate, atoms_by_id, cfg)

    def test_one_generator_contributes_a_third_of_the_weight(self) -> None:
        score = self._score_with_n_generators(["discourse"])
        # duration_fit=0 (20_000 is the min edge), silence_ratio=0 (no
        # internal pause on a single atom), lexical_hit=0.
        expected = 0.15 * (1 / 3) + 0.15 * 1
        assert score == expected

    def test_three_generators_hits_the_cap_of_one(self) -> None:
        score = self._score_with_n_generators(["discourse", "coverage", "lexical"])
        expected = 0.15 * 1 + 0.15 * 1 + 0.10 * 1  # lexical also in generators here
        assert score == expected

    def test_five_generators_is_still_capped_at_one_not_five_thirds(self) -> None:
        score_three = self._score_with_n_generators(["discourse", "coverage", "lexical"])
        score_five = self._score_with_n_generators(
            ["discourse", "coverage", "lexical", "llm_pass", "extra"]
        )
        # five generators includes llm_pass (+0.40), so subtract that out to
        # compare the capped n_generators contribution directly.
        assert score_five - 0.40 == score_three


class TestSilenceRatioProxy:
    def test_known_pause_values_produce_the_expected_ratio(self) -> None:
        cfg = Config(min_duration_ms=1_000, max_duration_ms=100_000)
        # three atoms spanning 0-30_000ms total; internal gap between atom 0
        # and atom 1 is 5_000ms (atom1.pause_before_ms), internal gap between
        # atom 1 and atom 2 is 2_000ms (atom2.pause_before_ms). total
        # internal pause = 7_000ms out of a 30_000ms span -> ratio 7/30.
        atoms_by_id = {
            0: _atom(0, 0, 10_000, pause_before_ms=0, pause_after_ms=5_000),
            1: _atom(1, 15_000, 20_000, pause_before_ms=5_000, pause_after_ms=2_000),
            2: _atom(2, 22_000, 30_000, pause_before_ms=2_000, pause_after_ms=0),
        }
        candidate = _candidate(["discourse"], 0, 2, 0, 30_000)

        score = prefilter_stage._compute_pre_score(candidate, atoms_by_id, cfg)
        silence_ratio = 7_000 / 30_000
        expected_silence_term = 0.15 * (1 - silence_ratio)

        # isolate: subtract every other term's known contribution.
        other_terms = 0.40 * 0 + 0.20 * prefilter_stage._duration_fit(30_000, cfg) + 0.15 * (1 / 3) + 0.10 * 0
        assert score == other_terms + expected_silence_term

    def test_external_pauses_outside_the_span_are_not_counted(self) -> None:
        # a pause BEFORE the first spanned atom or AFTER the last spanned
        # atom must not be counted -- only the internal gap between atom 0
        # and atom 1 (a 1_000ms candidate) should register.
        atoms_by_id = {
            0: _atom(0, 0, 5_000, pause_before_ms=999_000, pause_after_ms=1_000),
            1: _atom(1, 6_000, 10_000, pause_before_ms=1_000, pause_after_ms=999_000),
        }
        from clipper.util.silence import compute_timing_silence_ratio

        ratio = compute_timing_silence_ratio(0, 1, atoms_by_id)
        assert ratio == 1_000 / 10_000  # span is 0..10_000ms, internal pause is 1_000ms

    def test_shared_gap_between_two_internal_atoms_is_not_double_counted(self) -> None:
        # atom0.pause_after_ms and atom1.pause_before_ms both describe the
        # SAME 3_000ms gap -- summing both would double count it.
        atoms_by_id = {
            0: _atom(0, 0, 5_000, pause_before_ms=0, pause_after_ms=3_000),
            1: _atom(1, 8_000, 12_000, pause_before_ms=3_000, pause_after_ms=0),
        }
        from clipper.util.silence import compute_timing_silence_ratio

        ratio = compute_timing_silence_ratio(0, 1, atoms_by_id)
        assert ratio == 3_000 / 12_000  # span 0..12_000ms, single 3_000ms internal gap


class TestSelectTopAndCapEnforcement:
    def test_only_the_top_judge_cap_by_score_survive_when_none_are_llm_pass(self) -> None:
        candidates = [
            _candidate(["discourse"], i, i, 0, 0, candidate_id=f"c{i}").model_copy(
                update={"pre_score": score}
            )
            for i, score in enumerate([0.9, 0.1, 0.5, 0.7, 0.3])
        ]

        kept = prefilter_stage._select_top_and_llm_pass(candidates, judge_cap=3)

        assert len(kept) == 3
        assert {c.pre_score for c in kept} == {0.9, 0.7, 0.5}

    def test_llm_pass_candidate_ranked_below_the_cutoff_is_still_kept(self) -> None:
        high_scorers = [
            _candidate(["discourse"], i, i, 0, 0, candidate_id=f"c{i}").model_copy(
                update={"pre_score": 0.9}
            )
            for i in range(5)
        ]
        # a deliberately terrible llm_pass candidate: even with +0.40 for
        # llm_pass membership, its other terms are all zero, keeping it well
        # below every high-scorer above.
        weak_llm_pass = _candidate(["llm_pass"], 99, 99, 0, 0, candidate_id="weak_llm_pass").model_copy(
            update={"pre_score": 0.05}
        )
        candidates = high_scorers + [weak_llm_pass]

        kept = prefilter_stage._select_top_and_llm_pass(candidates, judge_cap=3)

        assert len(kept) == 4  # exceeds judge_cap because llm_pass is always kept
        assert any(c.id == "weak_llm_pass" for c in kept)


class TestRunPrefilterStageWiring:
    def test_reads_writes_and_logs_via_real_file_io(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        atoms = [_atom(i, i * 10_000, i * 10_000 + 10_000) for i in range(6)]
        (run_dir / "atoms.json").write_text(json.dumps([a.model_dump() for a in atoms]))

        cfg = Config(min_duration_ms=10_000, max_duration_ms=50_000, judge_cap=2)

        candidates = [
            _candidate(["discourse"], 0, 1, 0, 20_000, candidate_id="a"),
            _candidate(["lexical"], 2, 3, 20_000, 40_000, candidate_id="b"),
            _candidate(["coverage"], 4, 5, 40_000, 60_000, candidate_id="c"),
        ]
        (run_dir / "candidates.json").write_text(
            json.dumps([c.model_dump() for c in candidates])
        )

        prefilter_stage.run_prefilter(run_dir, cfg)

        written = json.loads((run_dir / "candidates.json").read_text())
        assert len(written) == 2  # judge_cap=2, none are llm_pass
        assert all(c["pre_score"] > 0.0 for c in written)  # placeholder 0.0 replaced

        stats = read_stats(run_dir)
        assert stats["candidates"]["after_prefilter"] == 2

        events = read_events(run_dir, stage="prefilter")
        event_names = [e["event"] for e in events]
        assert "started" in event_names
        assert "prefiltered" in event_names
        prefiltered_event = next(e for e in events if e["event"] == "prefiltered")
        assert prefiltered_event["data"]["before"] == 3
        assert prefiltered_event["data"]["after"] == 2
        assert prefiltered_event["data"]["dropped"] == 1
        assert prefiltered_event["data"]["cap_exceeded_by_llm_pass"] is False

    def test_llm_pass_candidate_ranked_below_the_cutoff_is_still_kept_by_the_real_stage(
        self, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # two "great" candidates: mid-range duration (fit=1), 3 generators
        # (n_generators term capped at max), no internal pause (silence
        # term=1), lexical hit -- score = 0.2+0.15+0.15+0.10 = 0.6 each.
        great_atoms = [_atom(i, i * 10_000, i * 10_000 + 10_000) for i in range(6)]
        # a deliberately weak llm_pass candidate spanning two tiny atoms
        # separated by a huge internal pause: duration at the min edge
        # (fit=0), a single generator (0.15*(1/3)), and silence_ratio close
        # to 1 (silence term close to 0). even with the +0.40 llm_pass term,
        # total score (~0.453) is well below the two great candidates' 0.6.
        weak_llm_pass_atoms = [
            _atom(6, 60_000, 60_100, pause_before_ms=0, pause_after_ms=9_800),
            _atom(7, 69_900, 70_000, pause_before_ms=9_800, pause_after_ms=0),
        ]
        atoms = great_atoms + weak_llm_pass_atoms
        (run_dir / "atoms.json").write_text(json.dumps([a.model_dump() for a in atoms]))

        cfg = Config(min_duration_ms=10_000, max_duration_ms=50_000, judge_cap=1)

        candidates = [
            _candidate(["discourse", "coverage", "lexical"], 0, 2, 0, 30_000, candidate_id="great1"),
            _candidate(["discourse", "coverage", "lexical"], 3, 5, 30_000, 60_000, candidate_id="great2"),
            _candidate(["llm_pass"], 6, 7, 60_000, 70_000, candidate_id="weak_llm_pass"),
        ]
        (run_dir / "candidates.json").write_text(
            json.dumps([c.model_dump() for c in candidates])
        )

        prefilter_stage.run_prefilter(run_dir, cfg)

        written = json.loads((run_dir / "candidates.json").read_text())
        assert len(written) == 2  # judge_cap=1 top scorer + the always-kept llm_pass straggler
        assert any(c["id"] == "weak_llm_pass" for c in written)

        events = read_events(run_dir, stage="prefilter")
        prefiltered_event = next(e for e in events if e["event"] == "prefiltered")
        assert prefiltered_event["data"]["cap_exceeded_by_llm_pass"] is True
        assert prefiltered_event["data"]["after"] == 2
