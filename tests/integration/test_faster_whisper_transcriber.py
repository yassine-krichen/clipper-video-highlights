"""integration test for `FasterWhisperTranscriber` (spec section 4.2).

marked `slow`: it loads a real local whisper model and runs real inference
against `tests/fixtures/sample_3min.mp4`'s real extracted audio -- no
mocking, since this is local compute, not network i/o (spec section 13).
deselect with `pytest -m "not slow"` for a fast default run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clipper.media.ffmpeg import extract_audio
from clipper.transcribers.faster_whisper import FasterWhisperTranscriber

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_3min.mp4"


@pytest.mark.slow
class TestFasterWhisperTranscriberRealAudio:
    def test_transcribes_real_audio_with_sane_word_timings(self, tmp_path: Path) -> None:
        audio_path = tmp_path / "audio.wav"
        extract_audio(_FIXTURE, audio_path)

        transcript = FasterWhisperTranscriber().transcribe(audio_path, language=None)

        assert transcript.provider == "faster-whisper"
        assert len(transcript.words) > 0
        assert len(transcript.segments) > 0
        # known fixture duration is ~144s; allow slack for vad trimming.
        assert 120_000 <= transcript.duration_ms <= 150_000

        for word in transcript.words:
            assert word.start_ms >= 0
            assert word.end_ms >= word.start_ms
            assert word.end_ms <= transcript.duration_ms + 5_000
