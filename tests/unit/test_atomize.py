"""tests for `clipper/stages/atomize.py` (spec section 4.3).

fully deterministic pure logic -- every test builds `Word` lists by hand, no
real audio/transcription involved (except the last section, which exercises
`run_atomize`'s actual file i/o against a hand-built `Transcript`).
"""

from __future__ import annotations

from pathlib import Path

from clipper.config import Config
from clipper.stages import atomize as atomize_stage
from clipper.stages.atomize import (
    atomize_words,
    compute_pauses,
    merge_short_atoms,
    split_into_atoms,
    split_long_atoms,
)
from clipper.types import Atom, Transcript, Word
from clipper.util.log import read_events, read_stats


def _word(text: str, start_ms: int, end_ms: int) -> Word:
    return Word(text=text, start_ms=start_ms, end_ms=end_ms)


def _word_identity(word: Word) -> tuple[str, int, int]:
    return (word.text, word.start_ms, word.end_ms)


def _assert_no_atom_exceeds_max_duration_unless_unsplittable(atoms: list[Atom]) -> None:
    """rule invariant (spec bullet 2): no atom's duration exceeds
    `_MAX_ATOM_MS`, unless `_select_split_index` genuinely has no valid
    split point for it (either no internal pause at all, or every candidate
    gap would create a new unprotected short fragment) -- the documented,
    local, well-scoped terminal exception. this is stricter than "must be a
    single word": a multi-word atom can also be a legitimate exception if
    every one of its internal gaps would carve off an unprotected <3-word
    fragment."""
    max_atom_ms = atomize_stage._MAX_ATOM_MS
    for atom in atoms:
        duration_ms = atom.end_ms - atom.start_ms
        if duration_ms > max_atom_ms:
            assert atomize_stage._select_split_index(atom.words) is None, (
                f"atom {atom.id!r} is {duration_ms}ms (> {max_atom_ms}ms) but has "
                "a valid split point it could have been split at"
            )


def _assert_no_short_atom_unless_pause_protected(atoms: list[Atom]) -> None:
    """rule invariant (spec bullet 3): no atom has fewer than
    `_MIN_ATOM_WORDS` words, unless it's separated from its (only available)
    neighbour by a pause `>= _PAUSE_SPLIT_MS`, or there is no neighbour to
    merge with at all (a single-atom result)."""
    min_words = atomize_stage._MIN_ATOM_WORDS
    pause_threshold = atomize_stage._PAUSE_SPLIT_MS
    for atom in atoms:
        if len(atom.words) >= min_words:
            continue
        if len(atoms) == 1:
            continue
        is_protected = atom.pause_before_ms >= pause_threshold or atom.pause_after_ms >= pause_threshold
        assert is_protected, (
            f"atom {atom.id!r} has only {len(atom.words)} words and is not "
            f"protected by a >={pause_threshold}ms pause on either side "
            f"(pause_before_ms={atom.pause_before_ms}, pause_after_ms={atom.pause_after_ms})"
        )


def _assert_invariants(words: list[Word], atoms: list[Atom]) -> None:
    """all spec-mandated invariants, reusable across scenario tests that
    exercise real pipeline output (i.e. anything produced by
    `atomize_words`/`run_atomize`, as opposed to a single pipeline function
    tested in isolation against a hand-built partial state). every such
    scenario test must call this rather than hand-rolling a subset of these
    checks, so the max-duration/short-atom invariants can never be silently
    skipped for a given test again (see DECISIONS.md, round-2 fix report)."""
    # (1) total atom span equals transcript span.
    transcript_start = min(w.start_ms for w in words)
    transcript_end = max(w.end_ms for w in words)
    atom_start = min(a.start_ms for a in atoms)
    atom_end = max(a.end_ms for a in atoms)
    assert (atom_start, atom_end) == (transcript_start, transcript_end)

    # (2) ids are 0..n-1, no gaps.
    assert [a.id for a in atoms] == list(range(len(atoms)))

    # (3) every word belongs to exactly one atom.
    expected_identities = sorted(_word_identity(w) for w in words)
    actual_identities = sorted(_word_identity(w) for atom in atoms for w in atom.words)
    assert actual_identities == expected_identities

    # (4) no atom exceeds the 25s limit unless it's a genuine, local,
    # well-scoped exception (no valid split point exists for it).
    _assert_no_atom_exceeds_max_duration_unless_unsplittable(atoms)

    # (5) no atom has fewer than 3 words unless it's pause-protected (or
    # there's nowhere else for it to go).
    _assert_no_short_atom_unless_pause_protected(atoms)


class TestInvariants:
    def test_span_ids_and_word_coverage_hold_for_a_simple_transcript(self) -> None:
        words = [
            _word("hello", 0, 200),
            _word("world.", 210, 500),
            _word("this", 1300, 1500),
            _word("is", 1510, 1650),
            _word("fine.", 1660, 1900),
        ]

        atoms = atomize_words(words)

        _assert_invariants(words, atoms)


class TestSentencePunctuationSplitting:
    def test_sentence_final_punctuation_starts_a_new_atom(self) -> None:
        words = [
            _word("hello.", 0, 200),
            _word("world", 210, 400),
        ]

        atoms = split_into_atoms(words)

        assert len(atoms) == 2
        assert [w.text for w in atoms[0].words] == ["hello."]
        assert [w.text for w in atoms[1].words] == ["world"]

    def test_no_punctuation_and_small_gap_stays_in_the_same_atom(self) -> None:
        words = [
            _word("hello", 0, 200),
            _word("world", 210, 400),
        ]

        atoms = split_into_atoms(words)

        assert len(atoms) == 1
        assert [w.text for w in atoms[0].words] == ["hello", "world"]


class TestPauseSplitting:
    def test_699ms_gap_does_not_split(self) -> None:
        words = [
            _word("hello", 0, 200),
            _word("world", 899, 1000),
        ]

        atoms = split_into_atoms(words)

        assert len(atoms) == 1

    def test_700ms_gap_splits(self) -> None:
        words = [
            _word("hello", 0, 200),
            _word("world", 900, 1000),
        ]

        atoms = split_into_atoms(words)

        assert len(atoms) == 2


class TestMergeShortAtoms:
    def test_atom_with_one_or_two_words_merges_into_following_atom(self) -> None:
        # "hi there" (2 words, no punctuation) is followed by a small gap,
        # then a 3-word atom -- both should end up merged into one atom.
        words = [
            _word("hi", 0, 100),
            _word("there", 110, 300),
            _word("nice", 400, 500),
            _word("to", 510, 600),
            _word("meet", 610, 700),
        ]
        atoms = [
            Atom(
                id=0,
                start_ms=0,
                end_ms=300,
                text="hi there",
                words=words[:2],
                pause_before_ms=0,
                pause_after_ms=0,
            ),
            Atom(
                id=1,
                start_ms=400,
                end_ms=700,
                text="nice to meet",
                words=words[2:],
                pause_before_ms=0,
                pause_after_ms=0,
            ),
        ]

        merged = merge_short_atoms(atoms)

        assert len(merged) == 1
        assert [w.text for w in merged[0].words] == [
            "hi",
            "there",
            "nice",
            "to",
            "meet",
        ]

    def test_atom_with_exactly_3_words_does_not_merge(self) -> None:
        words_a = [_word("one", 0, 100), _word("two", 110, 200), _word("three", 210, 300)]
        words_b = [_word("four", 400, 500), _word("five", 510, 600), _word("six", 610, 700)]
        atoms = [
            Atom(
                id=0,
                start_ms=0,
                end_ms=300,
                text="one two three",
                words=words_a,
                pause_before_ms=0,
                pause_after_ms=0,
            ),
            Atom(
                id=1,
                start_ms=400,
                end_ms=700,
                text="four five six",
                words=words_b,
                pause_before_ms=0,
                pause_after_ms=0,
            ),
        ]

        merged = merge_short_atoms(atoms)

        assert len(merged) == 2

    def test_short_atom_separated_by_large_pause_does_not_merge(self) -> None:
        words_a = [_word("hi", 0, 100), _word("there", 110, 300)]
        words_b = [
            _word("nice", 1000, 1100),
            _word("to", 1110, 1200),
            _word("meet", 1210, 1300),
        ]
        atoms = [
            Atom(
                id=0,
                start_ms=0,
                end_ms=300,
                text="hi there",
                words=words_a,
                pause_before_ms=0,
                pause_after_ms=0,
            ),
            Atom(
                id=1,
                start_ms=1000,
                end_ms=1300,
                text="nice to meet",
                words=words_b,
                pause_before_ms=0,
                pause_after_ms=0,
            ),
        ]

        # gap between the two atoms is 1000 - 300 = 700ms >= 700ms threshold.
        merged = merge_short_atoms(atoms)

        assert len(merged) == 2
        assert [w.text for w in merged[0].words] == ["hi", "there"]

    def test_three_consecutive_one_word_atoms_merge_across_two_fixed_point_rounds(self) -> None:
        # three consecutive 1-word atoms, no gaps/punctuation forcing a
        # split between them: a single left-to-right pass merges atom0 into
        # atom1 (making a 2-word atom) and, separately in the same pass,
        # would already have moved past atom2 -- so a second round is
        # needed to notice the now-2-word atom is still short and merge
        # atom2 in too. exercises the `while changed:` loop's actual
        # multi-round behaviour (previously untested beyond a single round).
        words = [
            _word("one", 0, 100),
            _word("two", 110, 200),
            _word("three", 210, 300),
        ]
        atoms = [
            Atom(
                id=0,
                start_ms=0,
                end_ms=100,
                text="one",
                words=[words[0]],
                pause_before_ms=0,
                pause_after_ms=0,
            ),
            Atom(
                id=1,
                start_ms=110,
                end_ms=200,
                text="two",
                words=[words[1]],
                pause_before_ms=0,
                pause_after_ms=0,
            ),
            Atom(
                id=2,
                start_ms=210,
                end_ms=300,
                text="three",
                words=[words[2]],
                pause_before_ms=0,
                pause_after_ms=0,
            ),
        ]

        merged = merge_short_atoms(atoms)

        assert len(merged) == 1
        assert [w.text for w in merged[0].words] == ["one", "two", "three"]

    def test_last_atom_short_separated_by_large_pause_stays_unmerged(self) -> None:
        # last atom is short (1 word) but separated from the previous atom
        # by a >=700ms pause -- the backward merge should respect the same
        # pause exception every forward merge does, and leave it alone.
        words_a = [
            _word("one", 0, 100),
            _word("two", 110, 200),
            _word("three", 210, 300),
        ]
        words_b = [_word("bye", 1000, 1100)]
        atoms = [
            Atom(
                id=0,
                start_ms=0,
                end_ms=300,
                text="one two three",
                words=words_a,
                pause_before_ms=0,
                pause_after_ms=0,
            ),
            Atom(
                id=1,
                start_ms=1000,
                end_ms=1100,
                text="bye",
                words=words_b,
                pause_before_ms=0,
                pause_after_ms=0,
            ),
        ]

        # gap between the two atoms is 1000 - 300 = 700ms >= 700ms threshold.
        merged = merge_short_atoms(atoms)

        assert len(merged) == 2
        assert [w.text for w in merged[1].words] == ["bye"]

    def test_last_atom_short_with_nothing_following_merges_backward(self) -> None:
        words_a = [
            _word("one", 0, 100),
            _word("two", 110, 200),
            _word("three", 210, 300),
        ]
        words_b = [_word("bye", 400, 500)]
        atoms = [
            Atom(
                id=0,
                start_ms=0,
                end_ms=300,
                text="one two three",
                words=words_a,
                pause_before_ms=0,
                pause_after_ms=0,
            ),
            Atom(
                id=1,
                start_ms=400,
                end_ms=500,
                text="bye",
                words=words_b,
                pause_before_ms=0,
                pause_after_ms=0,
            ),
        ]

        merged = merge_short_atoms(atoms)

        assert len(merged) == 1
        assert [w.text for w in merged[0].words] == ["one", "two", "three", "bye"]


class TestSplitLongAtoms:
    def test_long_atom_splits_at_longest_internal_pause(self) -> None:
        # atom spans 0 -> 26000ms (> 25000ms). the gap between "two" and
        # "three" (10000ms) is the largest internal pause.
        words = [
            _word("one", 0, 1000),
            _word("two", 1100, 2000),
            _word("three", 12000, 13000),
            _word("four", 13100, 26000),
        ]
        atom = Atom(
            id=0,
            start_ms=0,
            end_ms=26000,
            text="one two three four",
            words=words,
            pause_before_ms=0,
            pause_after_ms=0,
        )

        split = split_long_atoms([atom])

        assert len(split) == 2
        assert [w.text for w in split[0].words] == ["one", "two"]
        assert [w.text for w in split[1].words] == ["three", "four"]
        # combined span matches the original.
        assert split[0].start_ms == atom.start_ms
        assert split[-1].end_ms == atom.end_ms

    def test_long_atom_with_no_internal_pause_is_left_as_is(self) -> None:
        # a single word spanning > 25000ms: nothing to split at.
        words = [_word("looooong", 0, 30000)]
        atom = Atom(
            id=0,
            start_ms=0,
            end_ms=30000,
            text="looooong",
            words=words,
            pause_before_ms=0,
            pause_after_ms=0,
        )

        split = split_long_atoms([atom])

        assert len(split) == 1
        assert split[0].words == words

    def test_splitting_at_the_single_longest_gap_is_not_always_enough(self) -> None:
        # milestone review repro: splitting at the single longest internal
        # gap (a -> b, 28,900ms) leaves the *other* half (b, c, d) at
        # 31,100ms -- still way over 25s. that half has its own internal
        # pause (b -> c is small, c -> d is 27,700ms) and must be split
        # again on its own terms, recursively, not left as a single
        # oversized fragment just because it doesn't contain THE largest
        # gap in the original atom.
        words = [
            _word("a", 0, 100),
            _word("b", 29_000, 29_100),
            _word("c", 31_100, 31_200),
            _word("d", 60_000, 60_100),
        ]
        atom = Atom(
            id=0,
            start_ms=0,
            end_ms=60_100,
            text="a b c d",
            words=words,
            pause_before_ms=0,
            pause_after_ms=0,
        )

        split = split_long_atoms([atom])

        _assert_no_atom_exceeds_max_duration_unless_unsplittable(split)
        assert [w.text for w in split[0].words] == ["a"]
        assert [w.text for w in split[1].words] == ["b", "c"]
        assert [w.text for w in split[2].words] == ["d"]
        # combined span still matches the original.
        assert split[0].start_ms == atom.start_ms
        assert split[-1].end_ms == atom.end_ms


class TestComputePauses:
    def test_first_atom_has_zero_pause_before(self) -> None:
        atoms = [
            Atom(
                id=0,
                start_ms=0,
                end_ms=300,
                text="a",
                words=[_word("a", 0, 300)],
                pause_before_ms=0,
                pause_after_ms=0,
            ),
            Atom(
                id=1,
                start_ms=1000,
                end_ms=1300,
                text="b",
                words=[_word("b", 1000, 1300)],
                pause_before_ms=0,
                pause_after_ms=0,
            ),
        ]

        computed = compute_pauses(atoms)

        assert computed[0].pause_before_ms == 0

    def test_last_atom_has_zero_pause_after(self) -> None:
        atoms = [
            Atom(
                id=0,
                start_ms=0,
                end_ms=300,
                text="a",
                words=[_word("a", 0, 300)],
                pause_before_ms=0,
                pause_after_ms=0,
            ),
            Atom(
                id=1,
                start_ms=1000,
                end_ms=1300,
                text="b",
                words=[_word("b", 1000, 1300)],
                pause_before_ms=0,
                pause_after_ms=0,
            ),
        ]

        computed = compute_pauses(atoms)

        assert computed[-1].pause_after_ms == 0

    def test_middle_atom_pauses_computed_from_neighbour_word_timings(self) -> None:
        atoms = [
            Atom(
                id=0,
                start_ms=0,
                end_ms=300,
                text="a",
                words=[_word("a", 0, 300)],
                pause_before_ms=0,
                pause_after_ms=0,
            ),
            Atom(
                id=1,
                start_ms=1000,
                end_ms=1300,
                text="b",
                words=[_word("b", 1000, 1300)],
                pause_before_ms=0,
                pause_after_ms=0,
            ),
            Atom(
                id=2,
                start_ms=2000,
                end_ms=2300,
                text="c",
                words=[_word("c", 2000, 2300)],
                pause_before_ms=0,
                pause_after_ms=0,
            ),
        ]

        computed = compute_pauses(atoms)

        assert computed[1].pause_before_ms == 1000 - 300
        assert computed[1].pause_after_ms == 2000 - 1300


class TestEndToEndScenario:
    def test_realistic_transcript_exercises_multiple_rules_together(self) -> None:
        words = [
            # sentence 1: normal length, ends with punctuation.
            _word("well.", 0, 300),
            # a short 2-word fragment that should merge into what follows.
            _word("so", 400, 500),
            _word("um", 510, 600),
            # continues on with a small gap (no punctuation yet).
            _word("i", 650, 700),
            _word("think", 710, 900),
            _word("that's", 910, 1100),
            _word("interesting.", 1110, 1400),
            # 700ms+ pause -> new atom.
            _word("really", 2200, 2400),
            _word("cool", 2410, 2600),
            _word("stuff.", 2610, 2800),
        ]

        # a deliberately long run of back-to-back-ish words (small internal
        # gaps, one large one) with no sentence punctuation until the very
        # end, forcing a >25s atom that must be split at its longest pause.
        # hand-picked absolute timings so the total span is unambiguous.
        long_run_start = 3500
        long_words = [
            _word("word0", long_run_start, long_run_start + 800),
            _word("word1", long_run_start + 900, long_run_start + 1700),
            _word("word2", long_run_start + 1800, long_run_start + 2600),
            _word("word3", long_run_start + 2700, long_run_start + 3500),
            _word("word4", long_run_start + 3600, long_run_start + 4400),
            # longest internal pause: 18000ms gap here.
            _word("word5", long_run_start + 22400, long_run_start + 23200),
            _word("word6", long_run_start + 23300, long_run_start + 24100),
            _word("word7", long_run_start + 24200, long_run_start + 25000),
            _word("word8", long_run_start + 25100, long_run_start + 25900),
            _word("word9.", long_run_start + 26000, long_run_start + 26800),
        ]

        all_words = words + long_words

        atoms = atomize_words(all_words)

        _assert_invariants(all_words, atoms)
        # the long run's total span exceeds 25s, so it must have been split
        # into more than one atom.
        long_span_ms = long_words[-1].end_ms - long_words[0].start_ms
        assert long_span_ms > 25_000
        assert any(a.end_ms - a.start_ms <= 25_000 for a in atoms)
        for atom in atoms:
            assert len(atom.words) >= 1


class TestSplitLongAtomsUnsplittableException:
    def test_atom_where_every_split_point_would_create_an_unprotected_fragment_is_left_unsplit(
        self,
    ) -> None:
        # milestone review repro (bug 1, re-verified post-redesign): this
        # atom is a single 5-word run (25,100ms, no punctuation/big-enough
        # gaps to split at the word level). every one of its 4 internal gaps
        # is under 700ms, and every one of them would carve off a fragment
        # under 3 words on at least one side -- so `_select_split_index` has
        # no valid candidate at all. this is now resolved directly by
        # `split_long_atoms` (no merge-back, no cycle, no alternation): the
        # atom is left as a single, local, over-25s exception, exactly the
        # kind `_select_split_index`/`_assert_no_atom_exceeds_max_duration_
        # unless_unsplittable` document as legitimate. previously (round 2)
        # this same outcome was reached via an alternating merge/split loop
        # with cycle detection settling on the merged state -- that
        # machinery no longer exists, and the outcome here is a direct,
        # single-pass result of `split_long_atoms` alone.
        words = [
            _word("a", 0, 100),
            _word("b", 200, 300),
            _word("c", 400, 500),
            _word("d", 800, 900),
            _word("e", 1000, 25_100),
        ]

        atoms = atomize_words(words)

        _assert_invariants(words, atoms)
        assert len(atoms) == 1
        assert [w.text for w in atoms[0].words] == ["a", "b", "c", "d", "e"]


class TestSplitLongAtomsDoesNotAbsorbUnrelatedContent:
    def test_benign_atom_preceding_an_unsplittable_tail_is_not_absorbed_into_a_giant_merged_atom(
        self,
    ) -> None:
        # reviewer-constructed adversarial repro for the round-2 regression:
        # a single (pre-split) atom made of a ~20s, 100-word benign block
        # immediately followed by an 8-word tail, with the *single largest*
        # internal gap sitting right before the very last word (690ms, just
        # under the 700ms pause-split threshold, so it doesn't get split off
        # into its own atom by `split_into_atoms`). cutting at that single
        # largest gap alone would carve off a 1-word, unprotected trailing
        # fragment -- exactly the round-2 bug: the old alternating
        # merge/split loop would repeatedly merge that fragment backward,
        # and because the *whole* atom (all 108 words) is what gets
        # reproduced by the merge each round, cycle detection settled on
        # that single, giant, ~25.7s atom -- silently absorbing the entire
        # benign block into it.
        #
        # with the redesigned `_select_split_index`, the too-small candidate
        # (last gap, 690ms) is rejected and the next-largest gap -- the
        # boundary between the benign block and the tail (650ms) -- is
        # chosen instead, since splitting there leaves both sides with >=3
        # words. the benign block must therefore survive fully intact as
        # its own atom, never touched by the tail's split/merge resolution.
        benign_words = [_word(f"benign{i}", i * 200, i * 200 + 150) for i in range(100)]
        benign_span_ms = benign_words[-1].end_ms - benign_words[0].start_ms
        assert benign_span_ms < 25_000  # sanity: benign block alone is compliant

        tail_start = benign_words[-1].end_ms + 650  # boundary gap: 650ms
        tail_words = []
        cursor = tail_start
        for i in range(8):
            gap_before_last = 690  # the single largest gap in the whole atom
            tail_words.append(_word(f"tail{i}", cursor, cursor + 100))
            cursor += 100 + (gap_before_last if i == 6 else 600)

        words = benign_words + tail_words
        # sanity: this reproduces the pre-fix trigger condition -- the whole
        # thing is one atom (every internal gap is under 700ms) whose total
        # span is over 25s.
        assert all(
            words[i + 1].start_ms - words[i].end_ms < 700 for i in range(len(words) - 1)
        )
        total_span_ms = words[-1].end_ms - words[0].start_ms
        assert total_span_ms > 25_000

        atoms = atomize_words(words)

        _assert_invariants(words, atoms)
        # the benign block must survive as its own atom, completely intact
        # -- not absorbed into a merged blob with the tail.
        benign_atoms = [a for a in atoms if a.words[0].text == "benign0"]
        assert len(benign_atoms) == 1
        assert [w.text for w in benign_atoms[0].words] == [w.text for w in benign_words]
        assert benign_atoms[0].end_ms - benign_atoms[0].start_ms <= 25_000
        # no atom in the result may be a giant blob spanning both the
        # benign block and the tail.
        for atom in atoms:
            assert not (
                atom.words[0].text.startswith("benign") and atom.words[-1].text.startswith("tail")
            )


class TestRunAtomizeStage:
    def test_reads_transcript_and_writes_atoms_with_events_and_stats(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)

        words = [
            _word("hello", 0, 200),
            _word("world.", 210, 500),
            _word("this", 1300, 1500),
            _word("is", 1510, 1650),
            _word("fine.", 1660, 1900),
        ]
        transcript = Transcript(
            language="en",
            duration_ms=1900,
            words=words,
            segments=[],
            provider="freellm/groq",
            model="stub-model",
        )
        (run_dir / "transcript.json").write_text(transcript.model_dump_json())

        atomize_stage.run_atomize(run_dir, Config())

        atoms_path = run_dir / "atoms.json"
        assert atoms_path.exists()

        import json

        raw_atoms = json.loads(atoms_path.read_text())
        atoms = [Atom.model_validate(a) for a in raw_atoms]

        _assert_invariants(words, atoms)

        events = read_events(run_dir, stage="atomize")
        event_names = [e["event"] for e in events]
        assert "started" in event_names
        assert "atomized" in event_names

        stats = read_stats(run_dir)
        assert stats["stages"]["atomize"]["atom_count"] == len(atoms)
