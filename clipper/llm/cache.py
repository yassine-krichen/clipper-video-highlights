"""llm response cache (spec section 7): `sha256(model + prompt_version +
input)` keyed json blobs, stored both run-local and in a global,
cross-run cache so re-running a stage with unchanged inputs costs nothing.

cache lookup order (spec doesn't specify; decided and documented here, see
DECISIONS.md): run-local `run_dir/llm_cache/<key>.json` is checked first,
then the global `~/.clipper/cache/<key>.json`. a global-only hit is copied
into run-local too, so every run directory stays self-describing for later
inspection/debugging without needing to cross-reference the global cache.
"""

from __future__ import annotations

import json
from pathlib import Path

from clipper.util.atomic_write import atomic_write_json
from clipper.util.hashing import sha256_llm_cache_key

_GLOBAL_CACHE_DIR = Path.home() / ".clipper" / "cache"
_RUN_CACHE_DIRNAME = "llm_cache"


def cache_key(model: str, prompt_version: str, input_payload: str) -> str:
    """cache key for one llm call, per spec section 7's
    `sha256(model + prompt_version + input)` formula. thin wrapper over M2's
    `sha256_llm_cache_key` -- not reimplemented here."""
    return sha256_llm_cache_key(model, prompt_version, input_payload)


def _run_cache_path(run_dir: Path, key: str) -> Path:
    return run_dir / _RUN_CACHE_DIRNAME / f"{key}.json"


def _global_cache_path(key: str) -> Path:
    return _GLOBAL_CACHE_DIR / f"{key}.json"


def get_cached_response(run_dir: Path, key: str) -> dict | None:
    """look up a cached llm response by `key`: run-local first, then global.
    a global-only hit is copied into `run_dir`'s local cache before
    returning. returns `None` if `key` is in neither cache.

    no lock guards this read or the write-on-global-hit below: a given
    `key` is only ever written on a genuine cache miss (there's no path
    that overwrites an existing cache entry with different content), so two
    concurrent callers racing on the same key either both see a miss and
    both write the same response, or one sees the other's write -- there is
    no stale-clobbers-fresh scenario, unlike `stats.json`'s read-modify-write
    in `clipper/util/log.py`, which genuinely needs `record_stat`'s lock.
    this reasoning breaks if a future change makes the file i/o here
    non-blocking (e.g. wrapping it in `asyncio.to_thread` so concurrent
    judge-stage calls, spec section 7, don't block the event loop) --
    interleaved read-then-write across awaits could then race on the same
    key and would need the same kind of lock `record_stat` uses.
    """
    run_path = _run_cache_path(run_dir, key)
    if run_path.exists():
        return json.loads(run_path.read_text(encoding="utf-8"))

    global_path = _global_cache_path(key)
    if global_path.exists():
        data = json.loads(global_path.read_text(encoding="utf-8"))
        atomic_write_json(run_path, data)
        return data

    return None


def store_cached_response(run_dir: Path, key: str, response: dict) -> None:
    """write `response` to both the run-local and global caches, keyed by
    `key`. both writes are atomic."""
    atomic_write_json(_run_cache_path(run_dir, key), response)
    atomic_write_json(_global_cache_path(key), response)
