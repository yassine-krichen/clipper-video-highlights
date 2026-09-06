"""time formatting and run id helpers (spec section 2.2, 2.3, 4.11).

no llm calls, no stage logic -- pure functions only (milestone M2).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

_SLUG_MAX_LEN = 20


def format_ms_as_mmss(ms: int) -> str:
    """convert integer milliseconds to a `mm:ss` string, e.g. `2477200` ->
    `"41:17"` (spec section 4.11's `selected.md`).

    minutes are not capped at 59: a clip/run longer than an hour renders as
    e.g. `"61:01"` rather than rolling over into an `hh:mm:ss` format. the
    spec's examples never exceed an hour, so this is a deliberate,
    documented choice rather than a spec requirement.
    """
    total_seconds = ms // 1000
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def new_run_id(source_name: str) -> str:
    """generate a run id matching spec section 2.3's example format:
    `"2026-09-05T14-02-11_ep17"` -- an ISO-ish UTC timestamp (colons
    replaced with hyphens, since they're invalid in Windows paths) plus a
    filesystem-safe slug derived from the source filename. UTC rather than
    local time so run ids are stable and comparable regardless of the
    machine's timezone.
    """
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")

    stem = source_name.rsplit(".", 1)[0] if "." in source_name else source_name
    stem = stem.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]  # strip any directory parts
    slug = re.sub(r"[^a-z0-9]", "", stem.lower())[:_SLUG_MAX_LEN]

    return f"{timestamp}_{slug}" if slug else timestamp
