"""transcriber interface (spec section 4.2).

both `GroqTranscriber` (network, routed through freellmapi) and
`FasterWhisperTranscriber` (local, no network) implement this protocol so
`clipper/stages/transcribe.py` can select between them by config alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from clipper.types import Transcript


class Transcriber(Protocol):
    def transcribe(self, wav: Path, language: str | None) -> Transcript: ...
