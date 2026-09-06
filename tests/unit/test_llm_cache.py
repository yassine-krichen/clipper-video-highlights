"""tests for `clipper/llm/cache.py`. per spec section 13, this is pure local
i/o -- no mocking, exercised for real against `tmp_path`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipper.llm import cache


@pytest.fixture(autouse=True)
def _isolated_global_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """point the module's global cache dir at a tmp path so tests never
    touch the real `~/.clipper/cache/` on the dev machine."""
    global_dir = tmp_path / "global_cache"
    monkeypatch.setattr(cache, "_GLOBAL_CACHE_DIR", global_dir)
    return global_dir


class TestCacheKey:
    def test_matches_sha256_llm_cache_key(self) -> None:
        from clipper.util.hashing import sha256_llm_cache_key

        assert cache.cache_key("gemini-2.0-flash", "judge_v1", "hello") == (
            sha256_llm_cache_key("gemini-2.0-flash", "judge_v1", "hello")
        )


class TestGetCachedResponse:
    def test_returns_none_on_full_miss(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"

        assert cache.get_cached_response(run_dir, "does-not-exist") is None

    def test_returns_run_local_hit(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        cache.store_cached_response(run_dir, "key1", {"value": 1})

        assert cache.get_cached_response(run_dir, "key1") == {"value": 1}

    def test_global_only_hit_is_copied_into_run_local(
        self, tmp_path: Path, _isolated_global_cache: Path
    ) -> None:
        run_dir = tmp_path / "run"
        global_path = _isolated_global_cache / "key2.json"
        global_path.parent.mkdir(parents=True, exist_ok=True)
        global_path.write_text(json.dumps({"value": 2}), encoding="utf-8")

        result = cache.get_cached_response(run_dir, "key2")

        assert result == {"value": 2}
        run_local_path = run_dir / "llm_cache" / "key2.json"
        assert run_local_path.exists()
        assert json.loads(run_local_path.read_text(encoding="utf-8")) == {"value": 2}


class TestStoreCachedResponse:
    def test_writes_to_both_run_local_and_global(
        self, tmp_path: Path, _isolated_global_cache: Path
    ) -> None:
        run_dir = tmp_path / "run"

        cache.store_cached_response(run_dir, "key3", {"value": 3})

        run_local_path = run_dir / "llm_cache" / "key3.json"
        global_path = _isolated_global_cache / "key3.json"
        assert json.loads(run_local_path.read_text(encoding="utf-8")) == {"value": 3}
        assert json.loads(global_path.read_text(encoding="utf-8")) == {"value": 3}

    def test_round_trip_survives_store_then_get(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        payload = {"nested": {"a": 1, "b": [1, 2, 3]}, "s": "text"}

        cache.store_cached_response(run_dir, "key4", payload)

        assert cache.get_cached_response(run_dir, "key4") == payload
