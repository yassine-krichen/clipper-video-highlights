"""tests for `clipper/stages/refine.py` (spec section 4.8)."""

from __future__ import annotations

from clipper.config import Config
from clipper.stages.refine import refine_candidate, starts_with_dependent_word
from clipper.types import Atom
from tests.unit.builders import make_atom, make_judged

_CFG = Config(min_duration_ms=10_000, max_duration_ms=30_000)


def _atoms(texts: list[str], length_ms: int = 10_000, pause_ms: int = 1_000) -> dict[int, Atom]:
    atoms = []
    cursor = 0
    for i, text in enumerate(texts):
        start = cursor + (pause_ms if i else 0)
        atoms.append(
            make_atom(
                i,
                start,
                start + length_ms,
                text,
                pause_before_ms=pause_ms if i else 0,
                pause_after_ms=pause_ms if i < len(texts) - 1 else 0,
            )
        )
        cursor = start + length_ms
    return {a.id: a for a in atoms}


class TestStartsWithDependentWord:
    def test_flags_so_and_because_but_not_ordinary_openers(self) -> None:
        assert starts_with_dependent_word("So we shipped it")
        assert starts_with_dependent_word("Because nobody asked.")
        assert not starts_with_dependent_word("Sorry about that")
        assert not starts_with_dependent_word("That's a great story")

    def test_flags_thats_why_phrase(self) -> None:
        assert starts_with_dependent_word("That's why it failed")


class TestRefineCandidate:
    def test_adds_lead_in_and_tail_capped_at_half_the_adjacent_pause(self) -> None:
        atoms = _atoms(["a one", "b two", "c three"], pause_ms=1_000)
        judged = make_judged("c1", 1, 1, atoms[1].start_ms, atoms[1].end_ms)

        refined = refine_candidate(judged, atoms, 2, atoms[2].end_ms, _CFG)

        # pauses of 1000ms -> half is 500, capped at 300 lead-in and 400 tail.
        assert refined.start_ms == atoms[1].start_ms - 300
        assert refined.end_ms == atoms[1].end_ms + 400
        assert refined.segments[0].start_ms == refined.start_ms

    def test_applies_judge_suggestion_within_two_atoms(self) -> None:
        atoms = _atoms(["a", "b", "c", "d", "e", "f"], length_ms=10_000)
        judged = make_judged("c1", 2, 3, atoms[2].start_ms, atoms[3].end_ms, suggested=(1, 3))

        wide = Config(min_duration_ms=10_000, max_duration_ms=100_000)

        refined = refine_candidate(judged, atoms, 5, atoms[5].end_ms, wide)

        assert (refined.atom_start, refined.atom_end) == (1, 3)

    def test_clamps_a_suggestion_that_strays_more_than_two_atoms(self) -> None:
        atoms = _atoms(list("abcdefghij"), length_ms=10_000)
        judged = make_judged("c1", 5, 6, atoms[5].start_ms, atoms[6].end_ms, suggested=(0, 9))
        wide = Config(min_duration_ms=10_000, max_duration_ms=200_000)

        refined = refine_candidate(judged, atoms, 9, atoms[9].end_ms, wide)

        assert (refined.atom_start, refined.atom_end) == (3, 8)

    def test_drops_trailing_atom_when_too_long(self) -> None:
        atoms = _atoms(["a", "b", "c", "d"], length_ms=10_000)
        judged = make_judged("c1", 0, 3, 0, atoms[3].end_ms)

        refined = refine_candidate(judged, atoms, 3, atoms[3].end_ms, _CFG)

        assert refined.atom_start == 0
        assert refined.atom_end == 1
        assert any("dropped trailing atom" in line for line in refined.refinement_log)

    def test_adds_following_atom_when_too_short(self) -> None:
        atoms = _atoms(["a", "b", "c"], length_ms=6_000, pause_ms=500)
        judged = make_judged("c1", 0, 0, 0, atoms[0].end_ms)

        refined = refine_candidate(judged, atoms, 2, atoms[2].end_ms, _CFG)

        assert refined.atom_end == 1
        assert refined.end_ms - refined.start_ms >= _CFG.min_duration_ms

    def test_boundary_penalty_applies_per_dependent_edge_and_lowers_score(self) -> None:
        dependent_atoms = _atoms(["So that happened", "middle bit", "and then it ended"])
        clean_atoms = _atoms(["That happened", "middle bit", "it ended"])
        judged = make_judged("c1", 0, 2, 0, dependent_atoms[2].end_ms)
        cfg = Config(min_duration_ms=10_000, max_duration_ms=60_000)

        dependent = refine_candidate(judged, dependent_atoms, 2, dependent_atoms[2].end_ms, cfg)
        independent = refine_candidate(judged, clean_atoms, 2, clean_atoms[2].end_ms, cfg)

        assert dependent.penalties["boundary"] == 10.0
        assert independent.penalties["boundary"] == 0.0
        assert dependent.score < independent.score

    def test_never_expands_beyond_the_transcript(self) -> None:
        atoms = _atoms(["a", "b"], length_ms=4_000)
        judged = make_judged("c1", 0, 1, 0, atoms[1].end_ms)

        refined = refine_candidate(judged, atoms, 1, atoms[1].end_ms, _CFG)

        assert refined.start_ms >= 0
        assert refined.end_ms <= atoms[1].end_ms
        assert any("no atoms left" in line for line in refined.refinement_log)
