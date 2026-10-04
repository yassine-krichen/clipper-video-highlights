"""tests for `clipper/transcribers/groq.py`'s `GroqTranscriber`.

per spec section 13, only the network call (`audio.transcriptions.create`)
is mocked, with a recorded response fixture in `tests/fixtures/recorded/`.
the wav file itself is real (generated with ffmpeg's `lavfi` test source, as
`tests/integration/test_probe_stage.py` does elsewhere) since `chunk_wav`
calls real ffprobe on it -- ffmpeg/ffprobe are never mocked.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from openai.types.audio.transcription_verbose import TranscriptionVerbose

from clipper.transcribers.groq import GroqTranscriber, _build_client

_RECORDED_RESPONSE = (
    Path(__file__).parent.parent / "fixtures" / "recorded" / "groq_verbose_json_response.json"
)


def _make_short_wav(path: Path, duration_s: int = 5) -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration_s}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def _recorded_response() -> TranscriptionVerbose:
    return TranscriptionVerbose.model_validate(json.loads(_RECORDED_RESPONSE.read_text()))


@pytest.fixture
def _freellm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FREELLM_BASE_URL", "https://freellmapi.test/v1")
    monkeypatch.setenv("FREELLM_API_KEY", "test-key")


class TestBuildClientReadsEnv:
    def test_raises_when_base_url_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FREELLM_BASE_URL", raising=False)
        monkeypatch.setenv("FREELLM_API_KEY", "test-key")

        with pytest.raises(RuntimeError, match="FREELLM_BASE_URL"):
            _build_client()

    def test_raises_when_api_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FREELLM_BASE_URL", "https://freellmapi.test/v1")
        monkeypatch.delenv("FREELLM_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="FREELLM_API_KEY"):
            _build_client()

    def test_builds_client_pointed_at_freellmapi(self, _freellm_env: None) -> None:
        client = _build_client()

        assert str(client.base_url).rstrip("/") == "https://freellmapi.test/v1"
        assert client.api_key == "test-key"


class TestGroqTranscriberTranscribe:
    def test_parses_recorded_response_into_a_transcript(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _freellm_env: None,
        _recorded_response: TranscriptionVerbose,
    ) -> None:
        wav_path = tmp_path / "audio.wav"
        _make_short_wav(wav_path)

        transcriber = GroqTranscriber()
        monkeypatch.setattr(
            transcriber._client.audio.transcriptions,
            "create",
            lambda **kwargs: _recorded_response,
        )

        transcript = transcriber.transcribe(wav_path, language=None)

        assert transcript.provider == "freellm/groq"
        assert transcript.model == "whisper-large-v3-turbo"
        assert transcript.language == "english"
        assert transcript.duration_ms == 6500
        assert [w.text for w in transcript.words] == [
            "hello",
            "there",
            "world",
            "this",
            "is",
            "a",
            "test",
        ]
        assert transcript.words[0].start_ms == 0
        assert transcript.words[0].end_ms == 400
        assert transcript.words[2].start_ms == 1000
        assert len(transcript.segments) == 2
        assert transcript.segments[0]["text"] == "hello there world"

    def test_passes_language_through_when_set(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _freellm_env: None,
        _recorded_response: TranscriptionVerbose,
    ) -> None:
        wav_path = tmp_path / "audio.wav"
        _make_short_wav(wav_path)

        transcriber = GroqTranscriber()
        captured_kwargs: dict[str, object] = {}

        def _fake_create(**kwargs: object) -> TranscriptionVerbose:
            captured_kwargs.update(kwargs)
            return _recorded_response

        monkeypatch.setattr(
            transcriber._client.audio.transcriptions, "create", _fake_create
        )

        transcriber.transcribe(wav_path, language="en")

        assert captured_kwargs["language"] == "en"
        assert captured_kwargs["model"] == "whisper-large-v3-turbo"
        assert captured_kwargs["response_format"] == "verbose_json"
        assert captured_kwargs["timestamp_granularities"] == ["word", "segment"]


class TestWordsFromSegments:
    def test_spreads_segment_duration_over_words_by_length(self) -> None:
        from clipper.transcribers.groq import _words_from_segments

        words = _words_from_segments([{"start": 1.0, "end": 2.0, "text": " aa bbbb"}])

        assert [w.text for w in words] == ["aa", "bbbb"]
        assert words[0].start_ms == 1000
        assert words[0].end_ms == words[1].start_ms == 1333
        assert words[1].end_ms == 2000

    def test_segment_without_text_yields_no_words(self) -> None:
        from clipper.transcribers.groq import _words_from_segments

        assert _words_from_segments([{"start": 0.0, "end": 1.0, "text": "  "}]) == []
