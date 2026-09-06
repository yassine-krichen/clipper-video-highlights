"""pure unit tests for `clipper/transcribers/groq.py`'s `stitch_chunks`
(spec section 4.2).

no real audio, no network -- every transcript here is hand-built from
`Transcript`/`Word` directly, per the milestone's requirement that this
function be fully testable without either.
"""

from __future__ import annotations

import pytest

from clipper.transcribers.groq import stitch_chunks
from clipper.types import Transcript, Word


def _word(text: str, start_ms: int, end_ms: int) -> Word:
    return Word(text=text, start_ms=start_ms, end_ms=end_ms)


def _transcript(words: list[Word], duration_ms: int, segments: list[dict] | None = None) -> Transcript:
    return Transcript(
        language="en",
        duration_ms=duration_ms,
        words=words,
        segments=segments or [],
        provider="freellm/groq",
        model="whisper-large-v3-turbo",
    )


class TestStitchChunksDropsOverlapDuplicates:
    def test_drops_words_that_reappear_at_the_start_of_the_next_chunk(self) -> None:
        # chunk 0 covers [0, 600_000)ms of the source; its last two words
        # fall inside the [598_000, 600_000) overlap zone.
        chunk_0 = _transcript(
            words=[
                _word("hello", 597_000, 597_400),
                _word("world", 598_100, 598_500),  # in overlap zone
                _word("again", 599_000, 599_400),  # in overlap zone
            ],
            duration_ms=600_000,
        )
        # chunk 1 starts at absolute 598_000ms (step = 600_000 - 2_000). its
        # own clock starts at 0, so "world"/"again" reappear near its start,
        # timed the same (mod the offset) as chunk 0's tail -- these should
        # be dropped. "brand" is genuinely new and starts right at the edge
        # of the overlap zone but is not a timing duplicate, so it's kept.
        chunk_1 = _transcript(
            words=[
                _word("world", 100, 500),  # absolute 598_100-598_500: duplicate
                _word("again", 1_000, 1_400),  # absolute 599_000-599_400: duplicate
                _word("brand", 1_900, 2_300),  # absolute 599_900-600_300: new
                _word("new", 2_400, 2_800),  # absolute 600_400-600_800: new
            ],
            duration_ms=300_000,
        )

        result = stitch_chunks([chunk_0, chunk_1], [0, 598_000])

        assert [w.text for w in result.words] == ["hello", "world", "again", "brand", "new"]
        assert [(w.start_ms, w.end_ms) for w in result.words] == [
            (597_000, 597_400),
            (598_100, 598_500),
            (599_000, 599_400),
            (599_900, 600_300),
            (600_400, 600_800),
        ]

    def test_matches_on_timing_not_text(self) -> None:
        # the two chunks transcribe the same overlapping audio with
        # different text (e.g. one mis-hears a word) -- still deduplicated,
        # because the spec requires matching on timing, not text content.
        chunk_0 = _transcript(
            words=[_word("their", 598_200, 598_600)],
            duration_ms=600_000,
        )
        chunk_1 = _transcript(
            words=[_word("there", 200, 600)],  # absolute 598_200-598_600
            duration_ms=300_000,
        )

        result = stitch_chunks([chunk_0, chunk_1], [0, 598_000])

        assert len(result.words) == 1
        assert result.words[0].text == "their"  # first chunk's version wins


class TestStitchChunksThreeChunks:
    def test_dedupes_both_boundaries_across_three_chunks(self) -> None:
        # step = 600_000 - 2_000 = 598_000ms (chunk_wav's stride math), so
        # chunk 1 starts at absolute 598_000ms and chunk 2 starts at absolute
        # 1_196_000ms. chunk 0's tail overlaps chunk 1's start, and
        # (separately) chunk 1's own tail overlaps chunk 2's start -- this
        # guards `previous_tail` being recomputed from the full cumulative
        # `all_words` list (not just the immediately preceding chunk's own
        # word list) at each boundary, which a simpler "only look at the
        # previous chunk" implementation could still get right for N=2 but
        # would risk getting wrong once a third chunk's boundary needs its
        # own independent tail slice out of the cumulative timeline.
        chunk_0 = _transcript(
            words=[
                _word("hello", 597_000, 597_400),
                _word("world", 598_100, 598_500),  # in overlap zone w/ chunk 1
                _word("again", 599_000, 599_400),  # in overlap zone w/ chunk 1
            ],
            duration_ms=600_000,
        )
        chunk_1 = _transcript(
            words=[
                _word("world", 100, 500),  # absolute 598_100-598_500: dup of chunk 0
                _word("again", 1_000, 1_400),  # absolute 599_000-599_400: dup of chunk 0
                _word("middle", 5_000, 5_400),  # absolute 603_000-603_400: new
                _word("brand", 598_100, 598_500),  # absolute 1_196_100-1_196_500: in overlap zone w/ chunk 2
                _word("new", 599_000, 599_400),  # absolute 1_197_000-1_197_400: in overlap zone w/ chunk 2
            ],
            duration_ms=600_000,
        )
        chunk_2 = _transcript(
            words=[
                _word("brand", 100, 500),  # absolute 1_196_100-1_196_500: dup of chunk 1
                _word("new", 1_000, 1_400),  # absolute 1_197_000-1_197_400: dup of chunk 1
                _word("final", 1_900, 2_300),  # absolute 1_197_900-1_198_300: new
                _word("word", 2_400, 2_800),  # absolute 1_198_400-1_198_800: new
            ],
            duration_ms=300_000,
        )

        result = stitch_chunks([chunk_0, chunk_1, chunk_2], [0, 598_000, 1_196_000])

        assert [w.text for w in result.words] == [
            "hello",
            "world",
            "again",
            "middle",
            "brand",
            "new",
            "final",
            "word",
        ]
        starts = [w.start_ms for w in result.words]
        ends = [w.end_ms for w in result.words]
        assert starts == sorted(starts)  # monotonically increasing
        assert len(set(zip(starts, ends))) == len(result.words)  # no duplicated timings
        assert list(zip(starts, ends)) == [
            (597_000, 597_400),
            (598_100, 598_500),
            (599_000, 599_400),
            (603_000, 603_400),
            (1_196_100, 1_196_500),
            (1_197_000, 1_197_400),
            (1_197_900, 1_198_300),
            (1_198_400, 1_198_800),
        ]
        assert result.duration_ms == 1_496_000


class TestStitchChunksNoOverlapNeeded:
    def test_chunks_with_no_shared_timing_are_all_kept(self) -> None:
        chunk_0 = _transcript(words=[_word("hello", 0, 400)], duration_ms=600_000)
        chunk_1 = _transcript(words=[_word("world", 5_000, 5_400)], duration_ms=300_000)

        result = stitch_chunks([chunk_0, chunk_1], [0, 598_000])

        assert [w.text for w in result.words] == ["hello", "world"]
        assert result.words[1].start_ms == 603_000


class TestStitchChunksSingleChunk:
    def test_single_chunk_passes_through_unchanged(self) -> None:
        chunk = _transcript(
            words=[_word("hello", 0, 400), _word("world", 500, 900)],
            duration_ms=1_000,
            segments=[{"id": 0, "start": 0.0, "end": 1.0, "text": "hello world"}],
        )

        result = stitch_chunks([chunk], [0])

        assert result == chunk


class TestStitchChunksEmptyChunk:
    def test_chunk_with_no_words_contributes_nothing(self) -> None:
        chunk_0 = _transcript(words=[_word("hello", 0, 400)], duration_ms=600_000)
        empty_chunk = _transcript(words=[], duration_ms=300_000)

        result = stitch_chunks([chunk_0, empty_chunk], [0, 598_000])

        assert [w.text for w in result.words] == ["hello"]
        assert result.duration_ms == 898_000

    def test_raises_on_empty_chunk_list(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            stitch_chunks([], [])

    def test_raises_on_mismatched_lengths(self) -> None:
        chunk = _transcript(words=[_word("hello", 0, 400)], duration_ms=1_000)

        with pytest.raises(ValueError, match="same length"):
            stitch_chunks([chunk], [0, 1_000])


class TestStitchChunksShiftsSegments:
    def test_segment_start_end_are_shifted_by_offset(self) -> None:
        chunk_0 = _transcript(
            words=[_word("hello", 0, 400)],
            duration_ms=600_000,
            segments=[{"id": 0, "start": 0.0, "end": 0.4, "text": "hello"}],
        )
        chunk_1 = _transcript(
            words=[_word("world", 0, 400)],
            duration_ms=300_000,
            segments=[{"id": 0, "start": 0.0, "end": 0.4, "text": "world"}],
        )

        result = stitch_chunks([chunk_0, chunk_1], [0, 598_000])

        assert result.segments[0] == {"id": 0, "start": 0.0, "end": 0.4, "text": "hello"}
        assert result.segments[1] == {"id": 0, "start": 598.0, "end": 598.4, "text": "world"}
