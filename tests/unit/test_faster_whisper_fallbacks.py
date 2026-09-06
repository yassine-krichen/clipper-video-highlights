"""unit tests for `FasterWhisperTranscriber`'s two fallback branches (spec
section 4.2), using a fake/stub model injected via the `model_factory`
constructor param -- no real `WhisperModel`, no real gpu/cuda hardware, no
network. see `tests/integration/test_faster_whisper_transcriber.py` for the
real-model, slow-marked counterpart.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clipper.transcribers import faster_whisper as faster_whisper_module
from clipper.transcribers.faster_whisper import FasterWhisperTranscriber


class _FakeWord:
    def __init__(self, word: str, start: float, end: float, probability: float = 1.0) -> None:
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


class _FakeSegment:
    def __init__(self, id: int, start: float, end: float, text: str, words: list[_FakeWord]) -> None:
        self.id = id
        self.start = start
        self.end = end
        self.text = text
        self.words = words


class _FakeInfo:
    def __init__(self, duration: float, language: str = "en") -> None:
        self.duration = duration
        self.language = language


class _StaticModel:
    """fake model returning the same fixed result on every `.transcribe()` call."""

    def __init__(self, segments: list[_FakeSegment], duration: float) -> None:
        self._segments = segments
        self._duration = duration
        self.calls: list[dict[str, object]] = []

    def transcribe(
        self, wav_path: str, *, language: str | None, word_timestamps: bool, vad_filter: bool
    ) -> tuple[object, _FakeInfo]:
        self.calls.append({"language": language, "vad_filter": vad_filter})
        return iter(self._segments), _FakeInfo(self._duration)


class _RaisingModel:
    """fake model whose `.transcribe()` always raises a given exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls = 0

    def transcribe(
        self, wav_path: str, *, language: str | None, word_timestamps: bool, vad_filter: bool
    ) -> tuple[object, _FakeInfo]:
        self.calls += 1
        raise self._exc


class _VadSensitiveModel:
    """fake model that returns zero words when `vad_filter=True` and real
    words when `vad_filter=False` -- simulates the broken-vad environment
    documented in DECISIONS.md."""

    def __init__(self, words: list[_FakeWord]) -> None:
        self._words = words
        self.calls: list[bool] = []

    def transcribe(
        self, wav_path: str, *, language: str | None, word_timestamps: bool, vad_filter: bool
    ) -> tuple[object, _FakeInfo]:
        self.calls.append(vad_filter)
        if vad_filter:
            return iter([]), _FakeInfo(duration=5.0)
        segment = _FakeSegment(0, 0.0, 5.0, "hello world", self._words)
        return iter([segment]), _FakeInfo(duration=5.0)


class _AlwaysEmptyModel:
    """fake model that returns zero words regardless of `vad_filter`."""

    def __init__(self) -> None:
        self.calls: list[bool] = []

    def transcribe(
        self, wav_path: str, *, language: str | None, word_timestamps: bool, vad_filter: bool
    ) -> tuple[object, _FakeInfo]:
        self.calls.append(vad_filter)
        return iter([]), _FakeInfo(duration=5.0)


@pytest.fixture
def _gpu_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(faster_whisper_module, "_gpu_available", lambda: True)


@pytest.fixture
def _gpu_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(faster_whisper_module, "_gpu_available", lambda: False)


class TestCudaRuntimeFallback:
    def test_retries_on_cpu_when_cuda_runtime_is_unloadable(
        self, tmp_path: Path, _gpu_present: None
    ) -> None:
        real_words = [_FakeWord("hello", 0.0, 0.4), _FakeWord("world", 0.5, 0.9)]
        real_segment = _FakeSegment(0, 0.0, 0.9, "hello world", real_words)
        cpu_model = _StaticModel([real_segment], duration=1.0)
        cuda_model = _RaisingModel(RuntimeError("could not load library cublas64_12.dll"))

        def model_factory(model_size: str, device: str) -> object:
            return cuda_model if device == "cuda" else cpu_model

        transcriber = FasterWhisperTranscriber(model_factory=model_factory)

        transcript = transcriber.transcribe(tmp_path / "audio.wav", language=None)

        assert cuda_model.calls == 1
        assert [w.text for w in transcript.words] == ["hello", "world"]
        assert transcriber._device == "cpu"
        assert transcriber._model_size == "small"

    def test_unrelated_runtime_error_is_not_swallowed(
        self, tmp_path: Path, _gpu_present: None
    ) -> None:
        cuda_model = _RaisingModel(RuntimeError("input audio file is corrupt"))

        def model_factory(model_size: str, device: str) -> object:
            return cuda_model

        transcriber = FasterWhisperTranscriber(model_factory=model_factory)

        with pytest.raises(RuntimeError, match="corrupt"):
            transcriber.transcribe(tmp_path / "audio.wav", language=None)


class TestVadFallback:
    def test_retries_without_vad_when_first_attempt_returns_zero_words(
        self, tmp_path: Path, _gpu_absent: None
    ) -> None:
        real_words = [_FakeWord("hello", 0.0, 0.4), _FakeWord("world", 0.5, 0.9)]
        model = _VadSensitiveModel(real_words)

        transcriber = FasterWhisperTranscriber(model_factory=lambda size, device: model)

        transcript = transcriber.transcribe(tmp_path / "audio.wav", language=None)

        assert model.calls == [True, False]
        assert [w.text for w in transcript.words] == ["hello", "world"]

    def test_returns_empty_transcript_without_looping_when_still_zero_words(
        self, tmp_path: Path, _gpu_absent: None
    ) -> None:
        model = _AlwaysEmptyModel()

        transcriber = FasterWhisperTranscriber(model_factory=lambda size, device: model)

        transcript = transcriber.transcribe(tmp_path / "audio.wav", language=None)

        # exactly one retry (vad_filter=True, then vad_filter=False) -- not
        # an infinite loop -- and the terminal result is an empty transcript,
        # not a raised error.
        assert model.calls == [True, False]
        assert transcript.words == []
