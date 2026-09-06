"""probe + extract audio, the first pipeline stage (spec section 4.1).

deviates from the generic `run_<stage>(run_dir: Path, cfg: Config) -> None`
signature (spec section 4) by taking `video_path` as an extra parameter --
see DECISIONS.md. every later stage reads its input from files earlier
stages already wrote into `run_dir`, but nothing upstream of `probe` has
written the source video's path anywhere yet, so it has to be supplied
directly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from clipper.config import Config
from clipper.media.ffmpeg import AudioExtractionError, extract_audio
from clipper.media.probe import MediaProbeError, probe_media
from clipper.util.hashing import sha256_file
from clipper.util.log import append_event, record_stat

_STAGE = "probe"
_MIN_SOURCE_DURATION_MS = 60_000  # spec section 4.1: reject if duration < 60s

_AUDIO_WAV_FILENAME = "audio.wav"
_INPUT_JSON_FILENAME = "input.json"


class InvalidSourceError(Exception):
    """raised when the source video fails probe validation: unreadable, no
    audio stream, or duration outside [min_source_duration, max_source_duration]."""


def _duration_ms_from_probe(probe_result: dict) -> int:
    """extract the container duration (seconds, as a string) from ffprobe's
    `format.duration` field and convert to integer milliseconds.

    raises `InvalidSourceError` if `format` or `format.duration` is missing --
    ffprobe can exit 0 and produce well-formed json that's still missing this
    field for unreadable or corrupt media, so this is a distinct failure mode
    from `MediaProbeError` (which covers ffprobe itself failing to run)."""
    duration_s_str = probe_result.get("format", {}).get("duration")
    if duration_s_str is None:
        raise InvalidSourceError(
            "ffprobe output missing format.duration -- unreadable or corrupt media"
        )
    return round(float(duration_s_str) * 1000)


def _has_audio_stream(probe_result: dict) -> bool:
    streams = probe_result.get("streams", [])
    return any(stream.get("codec_type") == "audio" for stream in streams)


def run_probe(run_dir: Path, cfg: Config, video_path: Path) -> None:
    """probe `video_path`, reject invalid sources, extract audio to
    `run_dir/audio.wav`, and write `run_dir/input.json` (spec section 4.1).

    raises `InvalidSourceError` if the source is unreadable, has no audio
    stream, or its duration falls outside [60s, cfg.max_source_duration_ms].
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    append_event(run_dir, _STAGE, "started", {"video_path": str(video_path)})

    wall_start = time.monotonic()
    try:
        probe_result = probe_media(video_path)
    except MediaProbeError as exc:
        append_event(run_dir, _STAGE, "rejected", {"reason": str(exc)})
        raise InvalidSourceError(f"could not probe {video_path}: {exc}") from exc

    has_audio = _has_audio_stream(probe_result)
    try:
        duration_ms = _duration_ms_from_probe(probe_result)
    except InvalidSourceError as exc:
        append_event(run_dir, _STAGE, "rejected", {"reason": str(exc)})
        raise

    append_event(
        run_dir,
        _STAGE,
        "probed",
        {"duration_ms": duration_ms, "has_audio": has_audio},
    )

    if not has_audio:
        reason = "no audio stream found"
        append_event(run_dir, _STAGE, "rejected", {"reason": reason})
        raise InvalidSourceError(reason)

    if duration_ms < _MIN_SOURCE_DURATION_MS:
        reason = f"duration_ms {duration_ms} is below the {_MIN_SOURCE_DURATION_MS}ms minimum"
        append_event(run_dir, _STAGE, "rejected", {"reason": reason})
        raise InvalidSourceError(reason)

    if duration_ms > cfg.max_source_duration_ms:
        reason = (
            f"duration_ms {duration_ms} exceeds the configured "
            f"{cfg.max_source_duration_ms}ms maximum"
        )
        append_event(run_dir, _STAGE, "rejected", {"reason": reason})
        raise InvalidSourceError(reason)

    audio_path = run_dir / _AUDIO_WAV_FILENAME
    try:
        extract_audio(video_path, audio_path)
    except AudioExtractionError as exc:
        append_event(run_dir, _STAGE, "extraction_failed", {"reason": str(exc)})
        raise
    wall_ms = round((time.monotonic() - wall_start) * 1000)
    append_event(run_dir, _STAGE, "audio_extracted", {"audio_path": str(audio_path)})

    sha256 = sha256_file(video_path)

    input_data = {
        "probe": probe_result,
        "sha256": sha256,
        "config": cfg.model_dump(),
        "source_path": str(video_path),
        "duration_ms": duration_ms,
    }
    (run_dir / _INPUT_JSON_FILENAME).write_text(json.dumps(input_data, indent=2))

    record_stat(
        run_dir,
        "stages",
        _STAGE,
        {"wall_ms": wall_ms, "duration_ms": duration_ms, "sha256": sha256},
    )
