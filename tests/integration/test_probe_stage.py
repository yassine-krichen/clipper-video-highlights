"""tests for clipper/stages/probe.py (spec section 4.1).

per spec section 13, ffmpeg/ffprobe are never mocked. the happy path uses
the committed real fixture (`tests/fixtures/sample_3min.mp4`, ~144s,
854x480 h264/aac). the reject-path tests generate genuinely real (but
synthetic) short media files with ffmpeg at test time -- this exercises
`run_probe`'s rejection logic against real ffprobe output, not a mock of
the tool under test.
"""

import json
import subprocess
from pathlib import Path

import pytest

from clipper.config import Config
from clipper.media.ffmpeg import AudioExtractionError
from clipper.media.probe import probe_media
from clipper.stages.probe import InvalidSourceError, _duration_ms_from_probe, run_probe
from clipper.util.hashing import sha256_file
from clipper.util.log import read_events, read_stats

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_3min.mp4"


def _generate_media(path: Path, *, duration_s: int, with_audio: bool) -> None:
    """generate a real, small media file with ffmpeg's `lavfi` test sources
    -- a real file the pipeline's ffprobe/ffmpeg calls actually read, just
    synthetic content rather than a recorded video."""
    args = ["ffmpeg", "-y"]
    if with_audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=1000:duration={duration_s}"]
    args += ["-f", "lavfi", "-i", f"color=c=black:s=64x64:d={duration_s}"]
    args += ["-shortest"] if with_audio else ["-an", "-shortest"]
    args += [str(path)]

    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


class TestRunProbeHappyPath:
    def test_writes_input_json_and_audio_wav(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        cfg = Config()

        run_probe(run_dir, cfg, _FIXTURE)

        input_json_path = run_dir / "input.json"
        audio_wav_path = run_dir / "audio.wav"
        assert input_json_path.exists()
        assert audio_wav_path.exists()

        input_data = json.loads(input_json_path.read_text())
        assert input_data["sha256"] == sha256_file(_FIXTURE)
        assert input_data["config"] == cfg.model_dump()
        assert input_data["source_path"] == str(_FIXTURE)
        # known fixture duration is ~144s; allow slack for ffprobe rounding.
        assert 143_000 <= input_data["duration_ms"] <= 145_000
        assert "probe" in input_data
        assert "format" in input_data["probe"]
        assert "streams" in input_data["probe"]

    def test_extracted_audio_is_16khz_mono_pcm(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_probe(run_dir, Config(), _FIXTURE)

        audio_probe = probe_media(run_dir / "audio.wav")
        audio_streams = [s for s in audio_probe["streams"] if s["codec_type"] == "audio"]

        assert len(audio_streams) == 1
        stream = audio_streams[0]
        assert stream["sample_rate"] == "16000"
        assert stream["channels"] == 1
        assert stream["codec_name"] == "pcm_s16le"

    def test_creates_run_dir_if_missing(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "nested" / "run"
        assert not run_dir.exists()

        run_probe(run_dir, Config(), _FIXTURE)

        assert run_dir.exists()

    def test_logs_and_records_stats(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_probe(run_dir, Config(), _FIXTURE)

        events = read_events(run_dir, stage="probe")
        event_names = [e["event"] for e in events]
        assert "started" in event_names
        assert "probed" in event_names
        assert "audio_extracted" in event_names
        assert "rejected" not in event_names

        probed_event = next(e for e in events if e["event"] == "probed")
        assert probed_event["data"]["has_audio"] is True
        assert 143_000 <= probed_event["data"]["duration_ms"] <= 145_000

        stats = read_stats(run_dir)
        probe_stats = stats["stages"]["probe"]
        assert probe_stats["wall_ms"] > 0
        assert 143_000 <= probe_stats["duration_ms"] <= 145_000
        assert probe_stats["sha256"] == sha256_file(_FIXTURE)


class TestRunProbeRejectsInvalidSource:
    def test_rejects_duration_below_60s(self, tmp_path: Path) -> None:
        short_video = tmp_path / "short.mp4"
        _generate_media(short_video, duration_s=5, with_audio=True)
        run_dir = tmp_path / "run"

        with pytest.raises(InvalidSourceError, match="below"):
            run_probe(run_dir, Config(), short_video)

        events = read_events(run_dir, stage="probe")
        assert any(e["event"] == "rejected" for e in events)
        assert not (run_dir / "audio.wav").exists()

    def test_rejects_missing_audio_stream(self, tmp_path: Path) -> None:
        video_only = tmp_path / "video_only.mp4"
        _generate_media(video_only, duration_s=65, with_audio=False)
        run_dir = tmp_path / "run"

        with pytest.raises(InvalidSourceError, match="no audio stream"):
            run_probe(run_dir, Config(), video_only)

        events = read_events(run_dir, stage="probe")
        assert any(e["event"] == "rejected" for e in events)
        assert not (run_dir / "audio.wav").exists()

    def test_rejects_duration_above_configured_max(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        cfg = Config(max_source_duration_ms=100_000)  # fixture is ~144s

        with pytest.raises(InvalidSourceError, match="exceeds"):
            run_probe(run_dir, cfg, _FIXTURE)

        events = read_events(run_dir, stage="probe")
        assert any(e["event"] == "rejected" for e in events)
        assert not (run_dir / "audio.wav").exists()

    def test_rejects_unreadable_source(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.mp4"
        run_dir = tmp_path / "run"

        with pytest.raises(InvalidSourceError):
            run_probe(run_dir, Config(), missing)

        events = read_events(run_dir, stage="probe")
        assert any(e["event"] == "rejected" for e in events)


class TestRunProbeExtractionFailure:
    def test_logs_extraction_failed_when_ffmpeg_cannot_write_output(
        self, tmp_path: Path
    ) -> None:
        """drives a genuine (unmocked) ffmpeg extraction failure: the source
        passes ffprobe validation (real, valid audio+video, >=60s), but
        `run_dir/audio.wav` is pre-created as a directory, so ffmpeg's real
        extraction command fails to open its output path ("Permission
        denied") and exits non-zero -- exercising `AudioExtractionError`
        without mocking ffmpeg or ffprobe."""
        video = tmp_path / "video.mp4"
        _generate_media(video, duration_s=65, with_audio=True)
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "audio.wav").mkdir()  # forces ffmpeg's output open to fail

        with pytest.raises(AudioExtractionError):
            run_probe(run_dir, Config(), video)

        events = read_events(run_dir, stage="probe")
        assert any(e["event"] == "probed" for e in events)  # got past ffprobe
        extraction_failed_events = [e for e in events if e["event"] == "extraction_failed"]
        assert len(extraction_failed_events) == 1
        assert "reason" in extraction_failed_events[0]["data"]


class TestDurationMsFromProbeMalformedOutput:
    """`_duration_ms_from_probe` is unit-tested directly against a
    hand-constructed dict here -- this is testing this parsing function's own
    contract, not mocking ffprobe (spec section 13's ban only covers
    ffmpeg/ffprobe subprocess calls)."""

    def test_raises_invalid_source_error_when_format_key_missing(self) -> None:
        with pytest.raises(InvalidSourceError, match="format.duration"):
            _duration_ms_from_probe({"streams": []})

    def test_raises_invalid_source_error_when_duration_key_missing(self) -> None:
        with pytest.raises(InvalidSourceError, match="format.duration"):
            _duration_ms_from_probe({"format": {}, "streams": []})


class TestRunProbeDurationBoundaries:
    def test_accepts_duration_of_exactly_60_seconds(self, tmp_path: Path) -> None:
        video = tmp_path / "exactly_60s.mp4"
        _generate_media(video, duration_s=60, with_audio=True)
        run_dir = tmp_path / "run"

        run_probe(run_dir, Config(), video)  # must not raise

        events = read_events(run_dir, stage="probe")
        assert not any(e["event"] == "rejected" for e in events)
        probed_event = next(e for e in events if e["event"] == "probed")
        assert probed_event["data"]["duration_ms"] == 60_000

    def test_accepts_duration_of_exactly_configured_max(self, tmp_path: Path) -> None:
        video = tmp_path / "exactly_max.mp4"
        _generate_media(video, duration_s=65, with_audio=True)
        probed_duration_ms = _duration_ms_from_probe(probe_media(video))
        cfg = Config(max_source_duration_ms=probed_duration_ms)
        run_dir = tmp_path / "run"

        run_probe(run_dir, cfg, video)  # must not raise

        events = read_events(run_dir, stage="probe")
        assert not any(e["event"] == "rejected" for e in events)
        probed_event = next(e for e in events if e["event"] == "probed")
        assert probed_event["data"]["duration_ms"] == probed_duration_ms
