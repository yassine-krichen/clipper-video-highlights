"""tests for `clipper/generators/discourse.py` (g2, spec section 4.5).

fully deterministic pure logic -- every test builds `Atom` lists by hand, no
real transcription/llm involved.
"""

from __future__ import annotations

from clipper.config import Config
from clipper.generators.discourse import generate_discourse
from clipper.types import Atom, Word


def _atom(atom_id: int, start_ms: int, end_ms: int, pause_before_ms: int = 0) -> Atom:
    return Atom(
        id=atom_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text=f"atom {atom_id}",
        words=[Word(text=f"atom{atom_id}", start_ms=start_ms, end_ms=end_ms)],
        pause_before_ms=pause_before_ms,
        pause_after_ms=0,
    )


class TestRegionSplitting:
    def test_a_single_region_with_no_large_pauses_produces_candidates_spanning_it(self) -> None:
        # 5 atoms, each 6s long, back-to-back (30s total) -- one region.
        atoms = [_atom(i, i * 6000, i * 6000 + 6000) for i in range(5)]
        cfg = Config(min_duration_ms=10_000, max_duration_ms=30_000)

        candidates = generate_discourse(atoms, cfg)

        assert candidates  # the whole-region run (0..4) is a valid duration
        assert any(c.atom_start == 0 and c.atom_end == 4 for c in candidates)
        assert all(c.generators == ["discourse"] for c in candidates)

    def test_a_pause_of_1500ms_or_more_starts_a_new_region(self) -> None:
        # two atoms separated by a >=1500ms pause never appear in the same
        # candidate, since a new region starts at the second atom.
        atoms = [
            _atom(0, 0, 20_000),
            _atom(1, 21_500, 41_500, pause_before_ms=1500),
        ]
        cfg = Config(min_duration_ms=15_000, max_duration_ms=25_000)

        candidates = generate_discourse(atoms, cfg)

        assert all(not (c.atom_start == 0 and c.atom_end == 1) for c in candidates)

    def test_a_pause_just_under_1500ms_does_not_start_a_new_region(self) -> None:
        atoms = [
            _atom(0, 0, 10_000),
            _atom(1, 11_499, 21_499, pause_before_ms=1499),
        ]
        cfg = Config(min_duration_ms=15_000, max_duration_ms=25_000)

        candidates = generate_discourse(atoms, cfg)

        assert any(c.atom_start == 0 and c.atom_end == 1 for c in candidates)


class TestDurationBounds:
    def test_runs_outside_duration_bounds_are_never_emitted(self) -> None:
        atoms = [_atom(i, i * 1000, i * 1000 + 1000) for i in range(3)]  # 3s total
        cfg = Config(min_duration_ms=10_000, max_duration_ms=90_000)

        candidates = generate_discourse(atoms, cfg)

        assert candidates == []

    def test_runs_may_start_and_end_anywhere_within_a_region(self) -> None:
        # a region with a valid-duration run buried in the middle, not
        # anchored to the region's own start.
        atoms = [
            _atom(0, 0, 1000),
            _atom(1, 1000, 11_000),  # 10s window starting here: [1, 1]
            _atom(2, 11_000, 12_000),
        ]
        cfg = Config(min_duration_ms=9_000, max_duration_ms=11_000)

        candidates = generate_discourse(atoms, cfg)

        assert any(c.atom_start == 1 and c.atom_end == 1 for c in candidates)


class TestCapAtFourCandidatesPerRegion:
    def test_never_emits_more_than_four_candidates_for_one_region(self) -> None:
        # 20 short atoms, wide duration bounds -- many valid-duration runs
        # exist in a single region.
        atoms = [_atom(i, i * 1000, i * 1000 + 1000) for i in range(20)]
        cfg = Config(min_duration_ms=2000, max_duration_ms=20_000)

        candidates = generate_discourse(atoms, cfg)

        assert len(candidates) <= 4

    def test_selection_includes_the_shortest_and_longest_valid_runs(self) -> None:
        atoms = [_atom(i, i * 1000, i * 1000 + 1000) for i in range(20)]
        cfg = Config(min_duration_ms=2000, max_duration_ms=20_000)

        candidates = generate_discourse(atoms, cfg)
        durations = [c.end_ms - c.start_ms for c in candidates]

        assert min(durations) == 2000  # shortest-valid: 2 atoms, 2000ms
        assert max(durations) == 20_000  # longest: the whole region


class TestSpeakerChangeNoOp:
    def test_speaker_is_always_none_in_phase_1_so_no_region_split_from_it(self) -> None:
        atoms = [_atom(i, i * 6000, i * 6000 + 6000) for i in range(3)]
        assert all(atom.speaker is None for atom in atoms)  # sanity check
        cfg = Config(min_duration_ms=5000, max_duration_ms=20_000)

        candidates = generate_discourse(atoms, cfg)

        # a single region (no pauses >=1500ms), so the whole span is a
        # candidate -- if speaker-change splitting somehow fired, this run
        # would have been broken up and this candidate wouldn't exist.
        assert any(c.atom_start == 0 and c.atom_end == 2 for c in candidates)
