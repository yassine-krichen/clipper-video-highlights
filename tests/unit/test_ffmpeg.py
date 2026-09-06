"""tests for clipper/media/ffmpeg.py (spec section 4.1).

per spec section 13, ffmpeg is never mocked. these exercise a real
subprocess call against a genuinely nonexistent input path.
"""

from pathlib import Path

import pytest

from clipper.media.ffmpeg import AudioExtractionError, extract_audio


class TestExtractAudio:
    def test_raises_audio_extraction_error_for_nonexistent_input(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.mp4"
        output = tmp_path / "audio.wav"

        with pytest.raises(AudioExtractionError):
            extract_audio(missing, output)

        assert not output.exists()
