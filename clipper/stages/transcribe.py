"""transcribe stage (spec section 4.2): `run_dir/audio.wav` ->
`run_dir/transcript.json`.

picks a transcriber by `cfg.transcriber`, with a fallback: per spec,
`FasterWhisperTranscriber` is "only used when transcriber: faster_whisper or
when groq returns a non-retryable error". see `_is_non_retryable_groq_error`
for what counts as non-retryable here -- see DECISIONS.md.
"""

from __future__ import annotations

import time
from pathlib import Path

from clipper.config import Config
from clipper.transcribers.base import Transcriber
from clipper.transcribers.faster_whisper import FasterWhisperTranscriber
from clipper.transcribers.groq import GroqTranscriber
from clipper.util.log import append_event, record_stat

_STAGE = "transcribe"
_AUDIO_WAV_FILENAME = "audio.wav"
_TRANSCRIPT_JSON_FILENAME = "transcript.json"

# substrings of an exception's message that mark it as non-retryable --
# config/auth problems that a retry (or, here, a groq->faster_whisper
# fallback within the *same* run) cannot fix by trying again. rate limits,
# timeouts, and 5xx server errors are deliberately excluded: those are
# transient and retrying the same provider is the right response, not
# silently falling back to a slower, lower-quality local model. this is a
# heuristic, not an exhaustive classification of the openai sdk's exception
# hierarchy -- see DECISIONS.md.
_NON_RETRYABLE_ERROR_MARKERS = (
    "FREELLM_BASE_URL",
    "FREELLM_API_KEY",
    "authentication",
    "unauthorized",
    "invalid_api_key",
    "invalid api key",
    "401",
    "403",
    "not_found_error",
    "model_not_found",
)


def _is_non_retryable_groq_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker.lower() in message for marker in _NON_RETRYABLE_ERROR_MARKERS)


def _select_transcriber(cfg: Config) -> Transcriber:
    if cfg.transcriber == "faster_whisper":
        return FasterWhisperTranscriber()
    return GroqTranscriber()


def run_transcribe(run_dir: Path, cfg: Config) -> None:
    """transcribe `run_dir/audio.wav` and write `run_dir/transcript.json`
    (spec section 4.2). raises whatever the selected transcriber raises for
    an error that isn't recognized as a non-retryable groq error (in which
    case it falls back to `FasterWhisperTranscriber` instead of raising).
    """
    append_event(run_dir, _STAGE, "started", {"transcriber": cfg.transcriber})

    audio_path = run_dir / _AUDIO_WAV_FILENAME
    language = None if cfg.language == "auto" else cfg.language

    wall_start = time.monotonic()
    fallback_used = False

    if cfg.transcriber == "faster_whisper":
        transcript = FasterWhisperTranscriber().transcribe(audio_path, language)
    else:
        try:
            transcript = GroqTranscriber().transcribe(audio_path, language)
        except Exception as exc:
            if not _is_non_retryable_groq_error(exc):
                append_event(run_dir, _STAGE, "failed", {"reason": str(exc)})
                raise
            fallback_used = True
            append_event(
                run_dir,
                _STAGE,
                "fallback_to_faster_whisper",
                {"reason": str(exc)},
            )
            transcript = FasterWhisperTranscriber().transcribe(audio_path, language)

    wall_ms = round((time.monotonic() - wall_start) * 1000)

    (run_dir / _TRANSCRIPT_JSON_FILENAME).write_text(transcript.model_dump_json(indent=2))

    append_event(
        run_dir,
        _STAGE,
        "transcribed",
        {
            "provider": transcript.provider,
            "model": transcript.model,
            "duration_ms": transcript.duration_ms,
            "word_count": len(transcript.words),
            "fallback_used": fallback_used,
        },
    )

    record_stat(
        run_dir,
        "stages",
        _STAGE,
        {
            "wall_ms": wall_ms,
            "provider": transcript.provider,
            "model": transcript.model,
            "fallback_used": fallback_used,
        },
    )
