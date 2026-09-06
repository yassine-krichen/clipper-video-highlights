"""local, no-network fallback transcriber (spec section 4.2).

used when `cfg.transcriber == "faster_whisper"`, or when `GroqTranscriber`
raises a non-retryable error (see `clipper/stages/transcribe.py`). runs
entirely on local compute -- no freellmapi routing, nothing to mock in
tests (spec section 13: only network calls are mocked).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import ctranslate2
from faster_whisper import WhisperModel

from clipper.types import Transcript, Word

_PROVIDER = "faster-whisper"
_GPU_MODEL = "large-v3-turbo"
_CPU_MODEL = "small"


def _gpu_available() -> bool:
    """gpu detection via ctranslate2 (the actual inference backend
    faster-whisper runs on), not `torch.cuda.is_available()` -- torch is
    only a transitive dependency here (sentence-transformers, per
    pyproject.toml) and its cuda build isn't necessarily the one installed,
    so it can't be trusted to reflect what faster-whisper can actually use.
    ctranslate2 exposes its own device count directly.

    this only reflects driver-level gpu presence (e.g. `nvidia-smi` finding
    a card), not a working cuda runtime -- see `_load_model` for why that
    distinction matters."""
    return ctranslate2.get_cuda_device_count() > 0


def _load_model(model_size: str, device: str) -> WhisperModel:
    return WhisperModel(model_size, device=device)


# substrings of a ctranslate2 `RuntimeError` message that indicate a missing
# native cuda runtime library, as opposed to a genuine transcription
# failure -- see `FasterWhisperTranscriber.transcribe`'s fallback for why
# this check exists at all.
_CUDA_RUNTIME_LOAD_ERROR_MARKERS = ("cublas", "cudnn", ".dll", ".so")


def _is_cuda_runtime_load_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _CUDA_RUNTIME_LOAD_ERROR_MARKERS)


class FasterWhisperTranscriber:
    """`Transcriber` implementation running `faster-whisper` locally.
    model size follows gpu availability (spec section 4.2): `large-v3-turbo`
    on gpu, `small` on cpu.

    `_gpu_available()` only reflects driver-level gpu presence (e.g.
    `nvidia-smi` finding a card) -- ctranslate2 doesn't actually touch the
    cuda runtime (cublas/cudnn) until the first real inference call, so a
    missing runtime library (e.g. `cublas64_12.dll` on a windows box without
    a full cuda toolkit install) only surfaces as a `RuntimeError` out of
    `transcribe()`, not out of `WhisperModel(...)` construction. `transcribe`
    catches that specific failure mode and retries once on the cpu model --
    see DECISIONS.md.

    `transcribe` also retries once with `vad_filter=False` if the first
    (spec-mandated `vad_filter=True`) attempt comes back with zero words for
    non-empty audio -- a defense against a broken vad in the current
    environment, not a normal code path. see DECISIONS.md.
    """

    def __init__(self, *, model_factory: Callable[[str, str], WhisperModel] = _load_model) -> None:
        """`model_factory` defaults to the real `_load_model` (constructs a
        real `WhisperModel`); tests inject a fake/stub factory instead, so
        the cuda/vad fallback branches below are exercisable without a real
        model or gpu -- see `tests/unit/test_faster_whisper_fallbacks.py`.
        """
        self._model_factory = model_factory
        self._model_size = _GPU_MODEL if _gpu_available() else _CPU_MODEL
        self._device = "cuda" if self._model_size == _GPU_MODEL else "cpu"
        self._model = self._model_factory(self._model_size, self._device)

    def _fall_back_to_cpu_model(self) -> None:
        self._model_size = _CPU_MODEL
        self._device = "cpu"
        self._model = self._model_factory(self._model_size, self._device)

    def transcribe(self, wav: Path, language: str | None) -> Transcript:
        try:
            transcript = self._transcribe_with_loaded_model(wav, language, vad_filter=True)
        except RuntimeError as exc:
            if self._device != "cuda" or not _is_cuda_runtime_load_error(exc):
                raise
            self._fall_back_to_cpu_model()
            transcript = self._transcribe_with_loaded_model(wav, language, vad_filter=True)

        # some onnxruntime/silero-vad builds silently produce a speech
        # probability that never crosses the vad's detection threshold for
        # audio that genuinely contains speech (observed on this platform:
        # `vad_filter=True` yields zero segments for a real, non-silent
        # recording that transcribes normally with `vad_filter=False`) --
        # see DECISIONS.md. rather than ship a silently empty transcript,
        # retry once with vad filtering disabled: whisper's own decoder
        # still skips genuine silence, it just loses the pre-filtering
        # speed/accuracy benefit vad_filter otherwise provides.
        if not transcript.words and transcript.duration_ms > 0:
            transcript = self._transcribe_with_loaded_model(wav, language, vad_filter=False)

        return transcript

    def _transcribe_with_loaded_model(
        self, wav: Path, language: str | None, *, vad_filter: bool
    ) -> Transcript:
        segments_iter, info = self._model.transcribe(
            str(wav),
            language=language,
            word_timestamps=True,
            vad_filter=vad_filter,
        )

        words: list[Word] = []
        segments: list[dict] = []
        for segment in segments_iter:
            segments.append(
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
            )
            for word in segment.words or []:
                words.append(
                    Word(
                        text=word.word,
                        start_ms=round(word.start * 1000),
                        end_ms=round(word.end * 1000),
                        confidence=word.probability,
                    )
                )

        duration_ms = round(info.duration * 1000)

        return Transcript(
            language=language or info.language,
            duration_ms=duration_ms,
            words=words,
            segments=segments,
            provider=_PROVIDER,
            model=self._model_size,
        )
