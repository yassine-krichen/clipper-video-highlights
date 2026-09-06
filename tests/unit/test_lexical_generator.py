"""tests for `clipper/generators/lexical.py` (g3, spec section 4.5).

fully deterministic pure logic -- every test builds `Atom` lists by hand, no
real transcription/llm involved.
"""

from __future__ import annotations

from clipper.config import Config
from clipper.generators.lexical import generate_lexical
from clipper.types import Atom, Word


def _atom(atom_id: int, text: str, start_ms: int, end_ms: int) -> Atom:
    return Atom(
        id=atom_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        words=[Word(text=text, start_ms=start_ms, end_ms=end_ms)],
        pause_before_ms=0,
        pause_after_ms=0,
    )


class TestHookPhraseMatching:
    def test_matches_a_literal_spec_listed_hook_phrase_case_insensitively(self) -> None:
        atoms = [
            _atom(0, "so this was my BIGGEST MISTAKE ever", 0, 3000),
            _atom(1, "filler content here", 3000, 13_000),
        ]
        cfg = Config(min_duration_ms=5000, max_duration_ms=20_000)

        candidates = generate_lexical(atoms, cfg)

        assert candidates
        assert all(c.atom_start == 0 for c in candidates)
        assert all(c.generators == ["lexical"] for c in candidates)

    def test_matches_another_spec_listed_hook_phrase(self) -> None:
        atoms = [
            _atom(0, "here's what happened next", 0, 3000),
            _atom(1, "and then more", 3000, 13_000),
        ]
        cfg = Config(min_duration_ms=5000, max_duration_ms=20_000)

        candidates = generate_lexical(atoms, cfg)

        assert any(c.atom_start == 0 for c in candidates)

    def test_atom_with_no_hook_phrase_produces_no_candidates(self) -> None:
        atoms = [_atom(0, "just an ordinary sentence", 0, 3000)]
        cfg = Config(min_duration_ms=1000, max_duration_ms=5000)

        candidates = generate_lexical(atoms, cfg)

        assert candidates == []

    def test_currency_heuristic_matches_a_dollar_amount(self) -> None:
        atoms = [
            _atom(0, "it cost me $500 to fix", 0, 3000),
            _atom(1, "filler content", 3000, 13_000),
        ]
        cfg = Config(min_duration_ms=5000, max_duration_ms=20_000)

        candidates = generate_lexical(atoms, cfg)

        assert any(c.atom_start == 0 for c in candidates)

    def test_currency_heuristic_matches_a_spelled_out_amount(self) -> None:
        atoms = [
            _atom(0, "it cost me 500 dollars to fix", 0, 3000),
            _atom(1, "filler content", 3000, 13_000),
        ]
        cfg = Config(min_duration_ms=5000, max_duration_ms=20_000)

        candidates = generate_lexical(atoms, cfg)

        assert any(c.atom_start == 0 for c in candidates)


class TestTwoWindowsPerHookAtom:
    def test_emits_a_min_duration_window_and_a_max_duration_window(self) -> None:
        atoms = [
            _atom(0, "the truth is complicated", 0, 1000),
            _atom(1, "filler a", 1000, 6000),
            _atom(2, "filler b", 6000, 11_000),
            _atom(3, "filler c", 11_000, 30_000),
        ]
        cfg = Config(min_duration_ms=5000, max_duration_ms=11_000)

        candidates = generate_lexical(atoms, cfg)

        durations = sorted(c.end_ms - c.start_ms for c in candidates)
        # min-duration variant: [0,1] -> 6000ms is the first window >=5000ms
        assert durations[0] == 6000
        # max-duration variant: extended as far as fits within 11_000ms -> [0,2]
        assert durations[-1] == 11_000

    def test_min_variant_is_skipped_when_min_duration_is_unreachable(self) -> None:
        # a single, short hook atom with nothing to extend into can never
        # reach min_duration_ms -- the min-duration variant is skipped. the
        # max-duration variant is unaffected by this (it only cares about
        # not exceeding max_duration_ms, so it still fires here, anchored on
        # the hook atom alone); this generator doesn't itself enforce
        # min_duration_ms on the max variant -- that's `clipper/stages/
        # generate.py`'s normalization's job (drop-outside-duration-bounds).
        atoms = [_atom(0, "i lost everything that day", 0, 1000)]
        cfg = Config(min_duration_ms=60_000, max_duration_ms=90_000)

        candidates = generate_lexical(atoms, cfg)

        assert len(candidates) == 1
        assert candidates[0].atom_start == 0
        assert candidates[0].atom_end == 0

    def test_max_variant_is_skipped_when_the_hook_atom_alone_exceeds_max_duration(self) -> None:
        # the hook atom alone already exceeds max_duration_ms, so the
        # max-duration variant has no valid window (not even the atom
        # itself fits) and is skipped. the min-duration variant still fires
        # here (the atom alone already satisfies min_duration_ms too, since
        # min < max) -- again, this generator doesn't enforce
        # max_duration_ms on the min variant itself; normalization does.
        atoms = [_atom(0, "i lost everything that day", 0, 100_000)]
        cfg = Config(min_duration_ms=5000, max_duration_ms=10_000)

        candidates = generate_lexical(atoms, cfg)

        assert len(candidates) == 1
        assert candidates[0].atom_start == 0
        assert candidates[0].atom_end == 0
