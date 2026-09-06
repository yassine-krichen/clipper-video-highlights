"""run observability: `log.jsonl` structured events and `stats.json` counters
(spec section 2.2, section 10).

kept as plain `dict`/`list[dict]` rather than pydantic models: both files are
free-form, additive key-value bags written incrementally by many independent
call sites across every stage, and `stats.json`'s shape (section 10) is a
nested dict of heterogeneous value types (ints, floats, strings, nested
per-stage dicts) that doesn't earn a fixed schema at this milestone.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

_LOG_FILENAME = "log.jsonl"
_STATS_FILENAME = "stats.json"

# per-stats.json-path locks, so `record_stat`'s read-modify-write is
# serialized within a process (spec section 7: the judge stage fires up to 8
# concurrent asyncio llm calls, each recording stats). different run
# directories get independent locks and don't contend with each other. this
# is an in-process lock, not a cross-process one -- see DECISIONS.md.
_stats_locks: dict[Path, threading.Lock] = {}
_stats_locks_meta_lock = threading.Lock()


def _lock_for_stats_path(stats_path: Path) -> threading.Lock:
    """return the (creating if needed) `threading.Lock` for `stats_path`,
    keyed by its resolved absolute path so relative/absolute spellings of
    the same file share a lock."""
    resolved = stats_path.resolve()
    with _stats_locks_meta_lock:
        lock = _stats_locks.get(resolved)
        if lock is None:
            lock = threading.Lock()
            _stats_locks[resolved] = lock
        return lock


def append_event(run_dir: Path, stage: str, event: str, data: dict[str, object]) -> None:
    """append one `{ts, stage, event, data}` json line to
    `run_dir/log.jsonl` (spec section 10), creating the file if needed.
    `ts` is added automatically as the current utc time, in iso format.

    always opens in append mode -- safe to call many times across a run,
    never truncates prior events.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "stage": stage,
        "event": event,
        "data": data,
    }
    log_path = run_dir / _LOG_FILENAME
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_events(run_dir: Path, stage: str | None = None) -> list[dict[str, object]]:
    """read all events from `run_dir/log.jsonl` in order, optionally
    filtered to a single `stage`. returns `[]` if the file doesn't exist."""
    log_path = run_dir / _LOG_FILENAME
    if not log_path.exists():
        return []

    events: list[dict[str, object]] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if stage is None or event.get("stage") == stage:
                events.append(event)
    return events


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


def record_stat(run_dir: Path, section: str, key: str, value: object) -> None:
    """merge one value into `stats.json` at `data[section][key]` (spec
    section 10), read-modify-write so other sections/keys already present
    are left untouched. creates the file/section/key path as needed.

    the read-modify-write is guarded by an in-process lock keyed on
    `stats.json`'s resolved path, and the write itself is atomic (temp file
    + `os.replace`) -- safe to call from many concurrent asyncio tasks
    within one `clipper run` process (spec section 7's up to 8 concurrent
    judge calls). see DECISIONS.md.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_path = run_dir / _STATS_FILENAME
    lock = _lock_for_stats_path(stats_path)

    with lock:
        stats = read_stats(run_dir)
        stats.setdefault(section, {})[key] = value

        fd, tmp_name = tempfile.mkstemp(
            dir=stats_path.parent, prefix=f".{_STATS_FILENAME}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(stats, indent=2))
            _replace_with_retry(tmp_name, stats_path)
        except BaseException:
            os.remove(tmp_name)
            raise


def read_stats(run_dir: Path) -> dict[str, object]:
    """read `run_dir/stats.json`, or `{}` if it doesn't exist yet."""
    stats_path = run_dir / _STATS_FILENAME
    if not stats_path.exists():
        return {}
    return json.loads(stats_path.read_text(encoding="utf-8"))
