"""tests for clipper/media/probe.py (spec section 4.1).

per spec section 13, ffprobe is never mocked. these exercise real subprocess
calls against the committed fixture and a genuinely nonexistent path.
"""

from pathlib import Path

import pytest

from clipper.media.probe import MediaProbeError, probe_media

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_3min.mp4"


class TestProbeMedia:
    def test_returns_dict_with_format_and_streams_for_real_file(self) -> None:
        result = probe_media(_FIXTURE)

        assert "format" in result
        assert "streams" in result
        assert isinstance(result["streams"], list)
        assert len(result["streams"]) >= 1

    def test_raises_media_probe_error_for_nonexistent_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.mp4"

        with pytest.raises(MediaProbeError):
            probe_media(missing)
