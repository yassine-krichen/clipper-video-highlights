"""tests for `clipper/transcribers/groq.py`'s `chunk_wav` (spec section 4.2).

per spec section 13, ffmpeg/ffprobe are never mocked -- these run `chunk_wav`
against the real audio extracted from `tests/fixtures/sample_3min.mp4` (via
M3's `extract_audio`). the real fixture is only ~144s, shorter than the
10-minute production default, so a much shorter `chunk_duration_ms` (30s) is
passed explicitly to actually exercise multi-chunk splitting -- see
DECISIONS.md.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from clipper.media.ffmpeg import extract_audio
from clipper.media.probe import probe_media
from clipper.transcribers.groq import chunk_wav

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_3min.mp4"


def _extracted_audio(tmp_path: Path) -> Path:
    audio_path = tmp_path / "audio.wav"
    extract_audio(_FIXTURE, audio_path)
    return audio_path


def _duration_ms(wav_path: Path) -> int:
    return round(float(probe_media(wav_path)["format"]["duration"]) * 1000)


class TestChunkWavShortAudio:
    def test_audio_shorter_than_chunk_duration_is_returned_unchanged(
        self, tmp_path: Path
    ) -> None:
        audio_path = _extracted_audio(tmp_path)

        chunks = chunk_wav(audio_path, chunk_duration_ms=600_000, overlap_ms=2_000)

        assert chunks == [(audio_path, 0)]


class TestChunkWavMultiChunkSplitting:
    def test_splits_into_expected_number_of_chunks(self, tmp_path: Path) -> None:
        audio_path = _extracted_audio(tmp_path)

        chunks = chunk_wav(audio_path, chunk_duration_ms=30_000, overlap_ms=2_000)

        # step = 30_000 - 2_000 = 28_000ms; chunks start at 0, 28_000, 56_000,
        # 84_000, 112_000 (fifth chunk covers up to 142_000, still short of
        # the ~144_000ms total) and a final chunk covering the remainder.
        assert len(chunks) >= 5
        for chunk_path, _ in chunks:
            assert chunk_path.exists()
            assert chunk_path != audio_path

    def test_chunk_offsets_are_strictly_increasing_and_start_at_zero(
        self, tmp_path: Path
    ) -> None:
        audio_path = _extracted_audio(tmp_path)

        chunks = chunk_wav(audio_path, chunk_duration_ms=30_000, overlap_ms=2_000)
        offsets = [offset_ms for _, offset_ms in chunks]

        assert offsets[0] == 0
        assert offsets == sorted(set(offsets))
        assert all(b - a == 28_000 for a, b in pairwise(offsets))

    def test_consecutive_chunks_overlap_by_the_requested_amount(
        self, tmp_path: Path
    ) -> None:
        audio_path = _extracted_audio(tmp_path)

        chunks = chunk_wav(audio_path, chunk_duration_ms=30_000, overlap_ms=2_000)

        for (_, offset_a), (_, offset_b) in pairwise(chunks):
            chunk_a_end_ms = offset_a + 30_000
            overlap_ms = chunk_a_end_ms - offset_b
            assert overlap_ms == 2_000

    def test_last_chunk_covers_up_to_the_true_end_of_the_audio(self, tmp_path: Path) -> None:
        audio_path = _extracted_audio(tmp_path)
        total_duration_ms = _duration_ms(audio_path)

        chunks = chunk_wav(audio_path, chunk_duration_ms=30_000, overlap_ms=2_000)
        last_chunk_path, last_offset_ms = chunks[-1]
        last_chunk_duration_ms = _duration_ms(last_chunk_path)

        assert last_offset_ms + last_chunk_duration_ms >= total_duration_ms - 50  # ffmpeg slack

    def test_each_chunk_file_has_the_expected_duration(self, tmp_path: Path) -> None:
        audio_path = _extracted_audio(tmp_path)

        chunks = chunk_wav(audio_path, chunk_duration_ms=30_000, overlap_ms=2_000)

        for chunk_path, _offset_ms in chunks[:-1]:  # every chunk but the last is full-length
            assert abs(_duration_ms(chunk_path) - 30_000) <= 100
