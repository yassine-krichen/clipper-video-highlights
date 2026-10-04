"""groq-hosted whisper transcriber, routed through freellmapi (spec section
4.2; see DECISIONS.md 2026-09-06 on freellmapi routing).

`GroqTranscriber` speaks the `openai` sdk's audio transcription endpoint
against `FREELLM_BASE_URL`/`FREELLM_API_KEY` -- freellmapi is described as an
openai-compatible proxy that routes by model name, so pointing the sdk at it
with model `whisper-large-v3-turbo` is expected to reach groq's whisper
transparently.

groq (and, presumably, freellmapi passing through to it) caps request size,
so audio longer than `_CHUNK_DURATION_MS` is split into overlapping chunks
(`chunk_wav`), each chunk is transcribed independently, and the results are
stitched back into one absolute timeline (`stitch_chunks`). both of those are
pure/file-local functions with no network calls, kept separate from
`GroqTranscriber.transcribe` (the only method that talks to freellmapi) so
the chunking/stitching math is independently unit-testable -- see
DECISIONS.md for the chunk_wav implementation choice (ffmpeg subprocess
calls, mirroring `clipper/media/ffmpeg.py`'s approach elsewhere in the
codebase).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI
from openai.types.audio.transcription_verbose import TranscriptionVerbose

from clipper.media.probe import probe_media
from clipper.types import Transcript, Word

_MODEL = "whisper-large-v3-turbo"
_PROVIDER = "freellm/groq"
_CHUNK_DURATION_MS = 600_000  # 10 minutes, spec section 4.2
_OVERLAP_MS = 2_000  # spec section 4.2
_STITCH_TOLERANCE_MS = 100  # spec: match dropped overlap words on timing, not text


class ChunkExtractionError(Exception):
    """raised when ffmpeg fails to cut a wav chunk out of the source wav."""


def _wav_duration_ms(wav_path: Path) -> int:
    probe_result = probe_media(wav_path)
    return round(float(probe_result["format"]["duration"]) * 1000)


def _extract_chunk(wav_path: Path, chunk_path: Path, start_ms: int, duration_ms: int) -> None:
    """cut `[start_ms, start_ms + duration_ms)` out of `wav_path` into
    `chunk_path`, re-encoded to the same 16khz mono pcm shape (cheap for a
    wav source, and avoids any stream-copy seek-accuracy edge cases).
    """
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_ms / 1000:.3f}",
            "-i",
            str(wav_path),
            "-t",
            f"{duration_ms / 1000:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(chunk_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ChunkExtractionError(
            f"ffmpeg exited with code {result.returncode} chunking {wav_path} "
            f"at [{start_ms}, {start_ms + duration_ms})ms: {result.stderr.strip()}"
        )


def chunk_wav(
    wav_path: Path,
    chunk_duration_ms: int = _CHUNK_DURATION_MS,
    overlap_ms: int = _OVERLAP_MS,
) -> list[tuple[Path, int]]:
    """split `wav_path` into `chunk_duration_ms`-long pieces with `overlap_ms`
    of overlap between consecutive chunks, writing each chunk to a new
    temp file. returns `(chunk_path, start_offset_ms)` pairs, where
    `start_offset_ms` is the chunk's start position in `wav_path`'s own
    timeline -- needed to re-align each chunk's (0-based) word timings back
    to absolute time after it's transcribed independently.

    audio no longer than `chunk_duration_ms` is returned unchanged as a
    single `(wav_path, 0)` chunk -- no copy, no ffmpeg call.

    caller owns cleanup of the chunk files (they live in one temp directory
    together, so `shutil.rmtree` on `chunks[0][0].parent` is enough) unless
    the input was short enough to skip chunking.

    raises `ValueError` if `overlap_ms >= chunk_duration_ms` -- that would
    make the stride between consecutive chunks' start offsets zero or
    negative, so the splitting loop below would never advance past
    `duration_ms` and spin forever.
    """
    if overlap_ms >= chunk_duration_ms:
        raise ValueError(
            f"overlap_ms ({overlap_ms}) must be less than chunk_duration_ms "
            f"({chunk_duration_ms}) -- otherwise consecutive chunks never advance"
        )

    duration_ms = _wav_duration_ms(wav_path)
    if duration_ms <= chunk_duration_ms:
        return [(wav_path, 0)]

    step_ms = chunk_duration_ms - overlap_ms
    tmp_dir = Path(tempfile.mkdtemp(prefix="clipper_groq_chunks_"))

    chunks: list[tuple[Path, int]] = []
    start_ms = 0
    index = 0
    try:
        while start_ms < duration_ms:
            this_chunk_duration_ms = min(chunk_duration_ms, duration_ms - start_ms)
            chunk_path = tmp_dir / f"chunk_{index:03d}.wav"
            _extract_chunk(wav_path, chunk_path, start_ms, this_chunk_duration_ms)
            chunks.append((chunk_path, start_ms))

            if start_ms + this_chunk_duration_ms >= duration_ms:
                break
            start_ms += step_ms
            index += 1
    except Exception:
        # a mid-loop `_extract_chunk` failure (e.g. chunk 3 of 5) would
        # otherwise leak `tmp_dir` and any chunks already written into it --
        # the caller's cleanup (`GroqTranscriber.transcribe`'s `finally`)
        # only runs once `chunk_wav` has already returned successfully.
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return chunks


def _shift_word(word: Word, offset_ms: int) -> Word:
    return Word(
        text=word.text,
        start_ms=word.start_ms + offset_ms,
        end_ms=word.end_ms + offset_ms,
        confidence=word.confidence,
    )


def _shift_segment(segment: dict, offset_ms: int) -> dict:
    """shift a raw provider segment dict's `start`/`end` fields (seconds,
    groq/openai's verbose_json shape) by `offset_ms`, leaving every other key
    untouched. segments are kept only for debugging (spec section 3), so
    this is a best-effort shift of the commonly-present time fields rather
    than a full provider-schema-aware transform."""
    shifted = dict(segment)
    offset_s = offset_ms / 1000
    if "start" in shifted and shifted["start"] is not None:
        shifted["start"] = shifted["start"] + offset_s
    if "end" in shifted and shifted["end"] is not None:
        shifted["end"] = shifted["end"] + offset_s
    return shifted


def stitch_chunks(
    chunk_transcripts: list[Transcript],
    chunk_start_offsets_ms: list[int],
    overlap_ms: int = _OVERLAP_MS,
) -> Transcript:
    """merge per-chunk `Transcript`s (each with times relative to its own
    chunk, starting at 0) into one `Transcript` with absolute timings.

    pure function: no i/o, no network -- every chunk's words/segments are
    shifted by its `start_offset_ms`, then for every chunk after the first,
    words that fall inside the overlap zone (`[offset_ms, offset_ms +
    overlap_ms)` in absolute time -- the tail of the previous chunk covers
    the same audio) are dropped if a word already emitted from the previous
    chunk matches its timing within `_STITCH_TOLERANCE_MS`. matching is on
    timing only, never on text (spec section 4.2 is explicit about this --
    the two chunks' transcriptions of the same overlapping audio need not
    agree on exact wording/casing).
    """
    if not chunk_transcripts:
        raise ValueError("stitch_chunks requires at least one chunk transcript")
    if len(chunk_transcripts) != len(chunk_start_offsets_ms):
        raise ValueError("chunk_transcripts and chunk_start_offsets_ms must be the same length")

    all_words: list[Word] = []
    all_segments: list[dict] = []

    for index, (transcript, offset_ms) in enumerate(
        zip(chunk_transcripts, chunk_start_offsets_ms)
    ):
        shifted_words = [_shift_word(w, offset_ms) for w in transcript.words]
        shifted_segments = [_shift_segment(s, offset_ms) for s in transcript.segments]

        if index == 0:
            all_words.extend(shifted_words)
            all_segments.extend(shifted_segments)
            continue

        overlap_start_ms = offset_ms
        overlap_end_ms = offset_ms + overlap_ms
        previous_tail = [w for w in all_words if w.end_ms > overlap_start_ms]

        for word in shifted_words:
            in_overlap_zone = word.start_ms < overlap_end_ms
            duplicates_previous = in_overlap_zone and any(
                abs(word.start_ms - prev.start_ms) <= _STITCH_TOLERANCE_MS
                and abs(word.end_ms - prev.end_ms) <= _STITCH_TOLERANCE_MS
                for prev in previous_tail
            )
            if duplicates_previous:
                continue
            all_words.append(word)

        all_segments.extend(shifted_segments)

    all_words.sort(key=lambda w: w.start_ms)

    first = chunk_transcripts[0]
    last_offset_ms = chunk_start_offsets_ms[-1]
    duration_ms = last_offset_ms + chunk_transcripts[-1].duration_ms

    return Transcript(
        language=first.language,
        duration_ms=duration_ms,
        words=all_words,
        segments=all_segments,
        provider=first.provider,
        model=first.model,
    )


def _build_client() -> OpenAI:
    """`openai` sdk client pointed at the self-hosted freellmapi proxy --
    every llm/transcription call in this codebase routes through it (spec
    section 4.2 + DECISIONS.md 2026-09-06), never groq's own api directly.
    """
    base_url = os.environ.get("FREELLM_BASE_URL")
    api_key = os.environ.get("FREELLM_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError(
            "FREELLM_BASE_URL and FREELLM_API_KEY must be set (see .env.example) "
            "-- GroqTranscriber routes every call through the freellmapi proxy"
        )
    return OpenAI(base_url=base_url, api_key=api_key)


def _words_from_segments(segments: list[dict]) -> list[Word]:
    """approximate word timings by spreading each segment's duration over its
    words in proportion to their length. used only when the proxy returns
    segments but drops `words` (observed with freellmapi, 2026-10-04). boundaries
    are then only as precise as the segment edges; see DECISIONS.md."""
    words: list[Word] = []
    for segment in segments:
        tokens = str(segment.get("text", "")).split()
        if not tokens:
            continue
        start_ms = round(float(segment["start"]) * 1000)
        end_ms = round(float(segment["end"]) * 1000)
        total_chars = sum(len(t) for t in tokens)
        cursor = float(start_ms)
        for token in tokens:
            share = (end_ms - start_ms) * len(token) / total_chars
            words.append(
                Word(text=token, start_ms=round(cursor), end_ms=round(cursor + share), confidence=None)
            )
            cursor += share
    return words


def _parse_transcription_response(response: TranscriptionVerbose) -> Transcript:
    words = [
        Word(
            text=w.word,
            start_ms=round(w.start * 1000),
            end_ms=round(w.end * 1000),
            confidence=None,  # groq/openai's verbose_json word schema has no per-word confidence
        )
        for w in (response.words or [])
    ]
    segments = [s.model_dump() for s in (response.segments or [])]
    if not words and segments:
        words = _words_from_segments(segments)

    return Transcript(
        language=response.language or "auto",
        duration_ms=round((response.duration or 0.0) * 1000),
        words=words,
        segments=segments,
        provider=_PROVIDER,
        model=_MODEL,
    )


class GroqTranscriber:
    """`Transcriber` implementation calling groq's whisper model through the
    freellmapi proxy (spec section 4.2). transcribes audio under
    `_CHUNK_DURATION_MS` in one call; longer audio is chunked and stitched.
    """

    def __init__(self) -> None:
        self._client = _build_client()

    def _transcribe_single(self, wav: Path, language: str | None) -> Transcript:
        with wav.open("rb") as audio_file:
            kwargs: dict[str, object] = {
                "model": _MODEL,
                "file": audio_file,
                "response_format": "verbose_json",
                "timestamp_granularities": ["word", "segment"],
            }
            if language is not None:
                kwargs["language"] = language
            response = self._client.audio.transcriptions.create(**kwargs)
        return _parse_transcription_response(response)

    def transcribe(self, wav: Path, language: str | None) -> Transcript:
        chunks = chunk_wav(wav)
        if len(chunks) == 1 and chunks[0][0] == wav:
            return self._transcribe_single(wav, language)

        tmp_dir = chunks[0][0].parent
        try:
            chunk_transcripts = [
                self._transcribe_single(chunk_path, language) for chunk_path, _ in chunks
            ]
            offsets = [offset_ms for _, offset_ms in chunks]
            return stitch_chunks(chunk_transcripts, offsets)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
