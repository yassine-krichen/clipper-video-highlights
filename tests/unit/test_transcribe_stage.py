"""tests for `clipper/stages/transcribe.py`'s orchestration logic: transcriber
selection by config, and the groq -> faster_whisper fallback (spec section
4.2).

the transcribers themselves are stubbed at the class level here -- this
tests the stage's wiring, not re-testing `GroqTranscriber`/
`FasterWhisperTranscriber` (covered by their own test files). this is
distinct from mocking ffmpeg/network calls inside a transcriber, which spec
section 13 forbids/restricts respectively.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clipper.config import Config
from clipper.stages import transcribe as transcribe_stage
from clipper.types import Transcript, Word
from clipper.util.log import read_events, read_stats


def _stub_transcript(provider: str) -> Transcript:
    return Transcript(
        language="en",
        duration_ms=1000,
        words=[Word(text="hi", start_ms=0, end_ms=200)],
        segments=[],
        provider=provider,
        model="stub-model",
    )


class _StubGroqTranscriber:
    def __init__(self, *, error: Exception | None = None) -> None:
        self._error = error

    def transcribe(self, wav: Path, language: str | None) -> Transcript:
        if self._error is not None:
            raise self._error
        return _stub_transcript("freellm/groq")


class _StubFasterWhisperTranscriber:
    def transcribe(self, wav: Path, language: str | None) -> Transcript:
        return _stub_transcript("faster-whisper")


def _touch_audio_wav(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "audio.wav").write_bytes(b"")


class TestRunTranscribeSelectsByConfig:
    def test_uses_groq_transcriber_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        _touch_audio_wav(run_dir)
        monkeypatch.setattr(
            transcribe_stage, "GroqTranscriber", lambda: _StubGroqTranscriber()
        )

        transcribe_stage.run_transcribe(run_dir, Config(transcriber="groq"))

        transcript = Transcript.model_validate_json((run_dir / "transcript.json").read_text())
        assert transcript.provider == "freellm/groq"

    def test_uses_faster_whisper_transcriber_when_configured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        _touch_audio_wav(run_dir)
        monkeypatch.setattr(
            transcribe_stage, "FasterWhisperTranscriber", _StubFasterWhisperTranscriber
        )

        transcribe_stage.run_transcribe(run_dir, Config(transcriber="faster_whisper"))

        transcript = Transcript.model_validate_json((run_dir / "transcript.json").read_text())
        assert transcript.provider == "faster-whisper"

    def test_writes_events_and_stats(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        _touch_audio_wav(run_dir)
        monkeypatch.setattr(
            transcribe_stage, "GroqTranscriber", lambda: _StubGroqTranscriber()
        )

        transcribe_stage.run_transcribe(run_dir, Config(transcriber="groq"))

        events = read_events(run_dir, stage="transcribe")
        event_names = [e["event"] for e in events]
        assert "started" in event_names
        assert "transcribed" in event_names

        stats = read_stats(run_dir)
        assert stats["stages"]["transcribe"]["provider"] == "freellm/groq"
        assert stats["stages"]["transcribe"]["fallback_used"] is False


class TestRunTranscribeGroqFallback:
    def test_falls_back_to_faster_whisper_on_non_retryable_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        _touch_audio_wav(run_dir)
        auth_error = RuntimeError("FREELLM_API_KEY must be set")
        monkeypatch.setattr(
            transcribe_stage, "GroqTranscriber", lambda: _StubGroqTranscriber(error=auth_error)
        )
        monkeypatch.setattr(
            transcribe_stage, "FasterWhisperTranscriber", _StubFasterWhisperTranscriber
        )

        transcribe_stage.run_transcribe(run_dir, Config(transcriber="groq"))

        transcript = Transcript.model_validate_json((run_dir / "transcript.json").read_text())
        assert transcript.provider == "faster-whisper"

        events = read_events(run_dir, stage="transcribe")
        assert any(e["event"] == "fallback_to_faster_whisper" for e in events)

        stats = read_stats(run_dir)
        assert stats["stages"]["transcribe"]["fallback_used"] is True

    def test_raises_and_does_not_fall_back_on_retryable_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        _touch_audio_wav(run_dir)
        rate_limit_error = RuntimeError("429 rate limit exceeded, please retry later")
        monkeypatch.setattr(
            transcribe_stage,
            "GroqTranscriber",
            lambda: _StubGroqTranscriber(error=rate_limit_error),
        )

        def _fail_if_called() -> None:
            raise AssertionError("faster_whisper fallback should not be used for retryable errors")

        monkeypatch.setattr(transcribe_stage, "FasterWhisperTranscriber", _fail_if_called)

        with pytest.raises(RuntimeError, match="429"):
            transcribe_stage.run_transcribe(run_dir, Config(transcriber="groq"))

        events = read_events(run_dir, stage="transcribe")
        assert any(e["event"] == "failed" for e in events)
        assert not (run_dir / "transcript.json").exists()


class TestRunTranscribeLanguage:
    def test_passes_none_for_auto_language(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        _touch_audio_wav(run_dir)
        captured: dict[str, object] = {}

        class _CapturingTranscriber:
            def transcribe(self, wav: Path, language: str | None) -> Transcript:
                captured["language"] = language
                return _stub_transcript("freellm/groq")

        monkeypatch.setattr(transcribe_stage, "GroqTranscriber", _CapturingTranscriber)

        transcribe_stage.run_transcribe(run_dir, Config(transcriber="groq", language="auto"))

        assert captured["language"] is None

    def test_passes_configured_language_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        _touch_audio_wav(run_dir)
        captured: dict[str, object] = {}

        class _CapturingTranscriber:
            def transcribe(self, wav: Path, language: str | None) -> Transcript:
                captured["language"] = language
                return _stub_transcript("freellm/groq")

        monkeypatch.setattr(transcribe_stage, "GroqTranscriber", _CapturingTranscriber)

        transcribe_stage.run_transcribe(run_dir, Config(transcriber="groq", language="fr"))

        assert captured["language"] == "fr"


def test_empty_transcript_fails_loudly_instead_of_flowing_downstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _EmptyTranscriber:
        def transcribe(self, wav: Path, language: str | None) -> Transcript:
            return _stub_transcript("faster-whisper").model_copy(update={"words": []})

    run_dir = tmp_path / "run"
    _touch_audio_wav(run_dir)
    monkeypatch.setattr(transcribe_stage, "FasterWhisperTranscriber", lambda: _EmptyTranscriber())

    with pytest.raises(RuntimeError, match="no words"):
        transcribe_stage.run_transcribe(run_dir, Config(transcriber="faster_whisper"))

    assert not (run_dir / "transcript.json").exists()
