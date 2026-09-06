"""ffmpeg wrapper: extract a 16khz mono pcm wav from a source video (spec
section 4.1).
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class AudioExtractionError(Exception):
    """raised when ffmpeg fails to run or exits non-zero while extracting
    audio."""


def extract_audio(video_path: Path, output_wav_path: Path) -> None:
    """run `ffmpeg -i <video_path> -vn -ac 1 -ar 16000 -c:a pcm_s16le -y
    <output_wav_path>` (spec section 4.1's exact command, plus `-y` so
    re-running this stage overwrites its own output per spec section 4).
    """
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(output_wav_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AudioExtractionError(
            f"ffmpeg exited with code {result.returncode} extracting audio from "
            f"{video_path}: {result.stderr.strip()}"
        )
