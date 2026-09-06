"""shared atomic-json-write helper: temp file + `os.replace`, so a crash or
antivirus interruption mid-write can never leave the target file truncated
or corrupted. used by `clipper/util/log.py` (`stats.json`) and
`clipper/llm/cache.py` (cached llm responses) -- factored out of both to
remove the near-verbatim duplication between them (see DECISIONS.md).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path


def _replace_with_retry(src: str, dst: Path, attempts: int = 5) -> None:
    """`os.replace(src, dst)`, retrying briefly on `PermissionError`.

    on windows, another process (antivirus, search indexer) can transiently
    hold a newly-created file open for scanning right after it's written,
    making `os.replace` fail with `WinError 5` even though nothing in this
    process is still using the file. a short retry loop rides that out
    without weakening the atomicity of the replace itself.
    """
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.01 * (attempt + 1))


def atomic_write_json(path: Path, data: object) -> None:
    """write `data` as json to `path`, atomically (temp file in `path`'s
    directory, then `os.replace`) -- a crash or antivirus interruption
    mid-write can never leave `path` truncated or corrupted. creates
    `path`'s parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))
        _replace_with_retry(tmp_name, path)
    except BaseException:
        os.remove(tmp_name)
        raise
