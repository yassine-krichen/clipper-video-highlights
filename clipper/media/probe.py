"""ffprobe wrapper: read raw media metadata without trusting the filename or
extension (spec section 4.1's explicit instruction).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


class MediaProbeError(Exception):
    """raised when ffprobe fails to run, exits non-zero, or produces output
    that isn't valid json -- covers unreadable/corrupt/missing files."""


def probe_media(path: Path) -> dict:
    """run `ffprobe -v quiet -print_format json -show_format -show_streams`
    on `path` and return the parsed json as a raw dict (spec section 4.1).

    never inspects `path`'s extension -- ffprobe reads the file's actual
    container/codec data, so a mislabeled or extensionless file is handled
    the same way as a correctly named one.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise MediaProbeError(
            f"ffprobe exited with code {result.returncode} for {path}: {result.stderr.strip()}"
        )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError(f"ffprobe produced unparseable output for {path}: {exc}") from exc
