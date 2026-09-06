"""tests for `clipper/generators/coverage.py` (g4, spec section 4.5).

fully deterministic pure logic -- every test builds `Atom`/`Candidate` lists
by hand, no real transcription/llm involved.
"""

from __future__ import annotations

from clipper.config import Config
from clipper.generators.coverage import generate_coverage
from clipper.types import Atom, Candidate, Word


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


def _existing_candidate(atom_start: int, atom_end: int, start_ms: int, end_ms: int) -> Candidate:
    return Candidate(
        id=f"stub_{atom_start}_{atom_end}",
        atom_start=atom_start,
        atom_end=atom_end,
        start_ms=start_ms,
        end_ms=end_ms,
        generators=["discourse"],
        generator_hint=None,
        pre_score=0.0,
    )


class TestGapFinding:
    def test_no_candidates_generated_when_every_atom_is_already_covered(self) -> None:
        atoms = [_atom(i, i * 2000, i * 2000 + 2000) for i in range(5)]
        existing = [_existing_candidate(0, 4, 0, 10_000)]
        cfg = Config(min_duration_ms=1000, max_duration_ms=20_000)

        candidates = generate_coverage(atoms, cfg, existing)

        assert candidates == []

    def test_emits_a_window_starting_at_an_uncovered_atom(self) -> None:
        atoms = [_atom(i, i * 2000, i * 2000 + 2000) for i in range(5)]
        # only atoms 0-1 are covered; 2, 3, 4 are not.
        existing = [_existing_candidate(0, 1, 0, 4000)]
        cfg = Config(min_duration_ms=2000, max_duration_ms=20_000)

        candidates = generate_coverage(atoms, cfg, existing)

        assert candidates
        assert candidates[0].atom_start == 2
        assert all(c.generators == ["coverage"] for c in candidates)

    def test_adjacent_uncovered_atoms_within_one_window_produce_one_candidate_not_several(
        self,
    ) -> None:
        # a gap of 3 uncovered atoms (2000ms each = 6000ms), and
        # min_duration_ms is small enough that one window covers the whole
        # gap -- greedily marking every atom inside that window as covered
        # means the gap coalesces into exactly one candidate.
        atoms = [_atom(i, i * 2000, i * 2000 + 2000) for i in range(6)]
        existing = [_existing_candidate(0, 1, 0, 4000), _existing_candidate(5, 5, 10_000, 12_000)]
        cfg = Config(min_duration_ms=5000, max_duration_ms=10_000)

        candidates = generate_coverage(atoms, cfg, existing)

        assert len(candidates) == 1
        assert candidates[0].atom_start == 2

    def test_a_gap_longer_than_one_window_produces_multiple_tiled_candidates(self) -> None:
        # 15 uncovered atoms (2000ms each), with a min_duration_ms small
        # enough that a single window can't cover the whole gap -- expect
        # more than one coverage candidate, tiling the gap. 15 is chosen so
        # each 3-atom (6000ms) window divides the gap evenly, leaving no
        # trailing atom too short to reach min_duration_ms on its own (see
        # the dedicated trailing-gap test below for that case).
        atoms = [_atom(i, i * 2000, i * 2000 + 2000) for i in range(15)]
        cfg = Config(min_duration_ms=5000, max_duration_ms=10_000)

        candidates = generate_coverage(atoms, cfg, [])

        assert len(candidates) > 1
        covered_ids: set[int] = set()
        for candidate in candidates:
            covered_ids.update(range(candidate.atom_start, candidate.atom_end + 1))
        assert covered_ids == {atom.id for atom in atoms}

    def test_trailing_gap_too_short_to_reach_min_duration_is_left_uncovered(self) -> None:
        atoms = [_atom(0, 0, 2000), _atom(1, 2000, 4000)]
        existing = [_existing_candidate(0, 0, 0, 2000)]
        cfg = Config(min_duration_ms=10_000, max_duration_ms=20_000)

        candidates = generate_coverage(atoms, cfg, existing)

        assert candidates == []
