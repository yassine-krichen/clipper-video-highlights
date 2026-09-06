"""tests for `clipper/llm/client.py`'s `complete_structured`.

per spec section 13, only the network call
(`AsyncOpenAI.chat.completions.create`) is mocked -- cache/filesystem i/o
runs for real against `tmp_path`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from clipper.llm import cache as llm_cache
from clipper.llm import client
from clipper.llm.client import LLMSchemaError, _make_strict_schema, complete_structured
from clipper.llm.registry import resolve_model_alias
from clipper.util.log import read_events, read_stats


class _TestResponse(BaseModel):
    value: int


class _NestedSubmodel(BaseModel):
    label: str
    weight: float


class _NestedTestResponse(BaseModel):
    name: str
    priority: int = 3
    submodel: _NestedSubmodel


class _ListSubmodel(BaseModel):
    tag: str
    score: float


class _ListFieldTestResponse(BaseModel):
    """mirrors the shape M7's generator response model will almost
    certainly have: a list of proposed candidate windows."""

    windows: list[_ListSubmodel]
    optional_windows: list[_ListSubmodel] | None = None


@pytest.fixture(autouse=True)
def _freellm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FREELLM_BASE_URL", "https://freellmapi.test/v1")
    monkeypatch.setenv("FREELLM_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def _isolated_global_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """point the global cache dir at a tmp path so tests never touch, or
    collide via, the real `~/.clipper/cache/` on the dev machine (and don't
    leak cached responses across test cases, which all share the same
    system/user prompt content)."""
    monkeypatch.setattr(llm_cache, "_GLOBAL_CACHE_DIR", tmp_path / "global_cache")


def _fake_completion(content: str, tokens_in: int = 10, tokens_out: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=tokens_in, completion_tokens=tokens_out),
    )


class TestCompleteStructuredCacheMissThenSuccess:
    @pytest.mark.asyncio
    async def test_parses_and_caches_a_valid_response(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        calls: list[dict] = []

        async def fake_create(**kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return _fake_completion(json.dumps({"value": 42}))

        monkeypatch.setattr(
            client,
            "_build_client",
            lambda: SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
            ),
        )

        result = await complete_structured(
            run_dir=run_dir,
            model_alias="gemini-flash",
            prompt_version="test_v1",
            system_prompt="system",
            user_prompt="user",
            response_model=_TestResponse,
            temperature=0.0,
        )

        assert result == _TestResponse(value=42)
        assert len(calls) == 1
        assert calls[0]["model"] == resolve_model_alias("gemini-flash")

        cache_files = list((run_dir / "llm_cache").glob("*.json"))
        assert len(cache_files) == 1
        assert json.loads(cache_files[0].read_text(encoding="utf-8")) == {"value": 42}

        stats = read_stats(run_dir)
        recorded = list(stats["llm_calls"].values())
        assert len(recorded) == 1
        assert recorded[0]["cache_hit"] is False
        assert recorded[0]["tokens_in"] == 10
        assert recorded[0]["tokens_out"] == 5
        assert recorded[0]["cost_usd"] > 0


class TestCompleteStructuredCacheHit:
    @pytest.mark.asyncio
    async def test_returns_cached_result_without_calling_the_api(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"

        call_count = 0

        async def fake_create(**kwargs: object) -> SimpleNamespace:
            nonlocal call_count
            call_count += 1
            return _fake_completion(json.dumps({"value": 1}))

        monkeypatch.setattr(
            client,
            "_build_client",
            lambda: SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
            ),
        )

        first = await complete_structured(
            run_dir=run_dir,
            model_alias="gemini-flash",
            prompt_version="test_v1",
            system_prompt="system",
            user_prompt="user",
            response_model=_TestResponse,
            temperature=0.0,
        )
        assert first == _TestResponse(value=1)
        assert call_count == 1

        second = await complete_structured(
            run_dir=run_dir,
            model_alias="gemini-flash",
            prompt_version="test_v1",
            system_prompt="system",
            user_prompt="user",
            response_model=_TestResponse,
            temperature=0.0,
        )

        assert second == _TestResponse(value=1)
        assert call_count == 1  # no second api call

        stats = read_stats(run_dir)
        recorded = list(stats["llm_calls"].values())
        assert len(recorded) == 2
        cache_hit_entries = [r for r in recorded if r["cache_hit"] is True]
        assert len(cache_hit_entries) == 1
        assert cache_hit_entries[0]["tokens_in"] == 0
        assert cache_hit_entries[0]["tokens_out"] == 0
        assert cache_hit_entries[0]["cost_usd"] == 0.0


class TestCompleteStructuredRetryThenSucceed:
    @pytest.mark.asyncio
    async def test_appends_validation_error_and_succeeds_on_second_attempt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        calls: list[dict] = []

        async def fake_create(**kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            if len(calls) == 1:
                return _fake_completion("not valid json", tokens_in=10, tokens_out=5)
            return _fake_completion(json.dumps({"value": 7}), tokens_in=20, tokens_out=8)

        monkeypatch.setattr(
            client,
            "_build_client",
            lambda: SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
            ),
        )

        result = await complete_structured(
            run_dir=run_dir,
            model_alias="gemini-flash",
            prompt_version="test_v1",
            system_prompt="system",
            user_prompt="user",
            response_model=_TestResponse,
            temperature=0.0,
        )

        assert result == _TestResponse(value=7)
        assert len(calls) == 2

        second_attempt_user_message = calls[1]["messages"][1]["content"]
        assert "user" in second_attempt_user_message
        assert "failed validation" in second_attempt_user_message

        events = read_events(run_dir, stage="llm")
        assert any(e["event"] == "retry" for e in events)

        stats = read_stats(run_dir)
        recorded = list(stats["llm_calls"].values())
        assert len(recorded) == 2  # one per real api call: the failed attempt + the success
        assert all(r["cache_hit"] is False for r in recorded)
        # distinct token values per attempt confirm each entry reflects its
        # own real api call rather than replaying the same/last usage object
        assert recorded[0]["tokens_in"] == 10
        assert recorded[0]["tokens_out"] == 5
        assert recorded[1]["tokens_in"] == 20
        assert recorded[1]["tokens_out"] == 8
        assert all(r["cost_usd"] > 0 for r in recorded)


class TestCompleteStructuredRetryExhausted:
    @pytest.mark.asyncio
    async def test_raises_llm_schema_error_after_exactly_three_attempts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        calls: list[dict] = []

        async def fake_create(**kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            attempt_tokens = [(10, 5), (20, 8), (30, 12)]
            tokens_in, tokens_out = attempt_tokens[len(calls) - 1]
            return _fake_completion("still not valid json", tokens_in=tokens_in, tokens_out=tokens_out)

        monkeypatch.setattr(
            client,
            "_build_client",
            lambda: SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
            ),
        )

        with pytest.raises(LLMSchemaError):
            await complete_structured(
                run_dir=run_dir,
                model_alias="gemini-flash",
                prompt_version="test_v1",
                system_prompt="system",
                user_prompt="user",
                response_model=_TestResponse,
                temperature=0.0,
            )

        assert len(calls) == 3

        events = read_events(run_dir, stage="llm")
        assert any(e["event"] == "call_failed" for e in events)

        # nothing should have been cached on total failure
        assert not (run_dir / "llm_cache").exists() or not list(
            (run_dir / "llm_cache").glob("*.json")
        )

        # every failed attempt made a real, billed api call -- all 3 must
        # still be recorded even though the whole thing ultimately failed
        stats = read_stats(run_dir)
        recorded = list(stats["llm_calls"].values())
        assert len(recorded) == 3
        assert all(r["cache_hit"] is False for r in recorded)
        # distinct token values per attempt confirm each entry reflects its
        # own real api call rather than replaying the same/last usage object
        assert recorded[0]["tokens_in"] == 10
        assert recorded[0]["tokens_out"] == 5
        assert recorded[1]["tokens_in"] == 20
        assert recorded[1]["tokens_out"] == 8
        assert recorded[2]["tokens_in"] == 30
        assert recorded[2]["tokens_out"] == 12
        assert all(r["cost_usd"] > 0 for r in recorded)


def _all_object_nodes(schema: dict) -> list[dict]:
    """every object-type schema node reachable from `schema`: itself (if an
    object) plus every entry under `$defs`. good enough for these
    structural assertions since none of the test models nest an object
    inside a list/anyOf."""
    nodes = []
    if "properties" in schema:
        nodes.append(schema)
    for def_node in schema.get("$defs", {}).values():
        if "properties" in def_node:
            nodes.append(def_node)
    return nodes


class TestMakeStrictSchema:
    def test_every_object_node_forbids_additional_properties(self) -> None:
        schema = _make_strict_schema(_NestedTestResponse.model_json_schema())

        object_nodes = _all_object_nodes(schema)
        assert len(object_nodes) == 2  # top-level Resp + nested Sub under $defs
        assert all(node["additionalProperties"] is False for node in object_nodes)

    def test_every_object_node_requires_all_of_its_own_properties(self) -> None:
        schema = _make_strict_schema(_NestedTestResponse.model_json_schema())

        for node in _all_object_nodes(schema):
            assert set(node["required"]) == set(node["properties"].keys())

    def test_originally_optional_field_type_now_allows_null(self) -> None:
        schema = _make_strict_schema(_NestedTestResponse.model_json_schema())

        priority_schema = schema["properties"]["priority"]
        assert priority_schema["type"] == ["integer", "null"]

    def test_originally_required_field_type_is_unchanged(self) -> None:
        schema = _make_strict_schema(_NestedTestResponse.model_json_schema())

        name_schema = schema["properties"]["name"]
        assert name_schema["type"] == "string"

    def test_nested_submodel_ref_still_resolves_via_defs(self) -> None:
        schema = _make_strict_schema(_NestedTestResponse.model_json_schema())

        submodel_node = schema["$defs"]["_NestedSubmodel"]
        assert submodel_node["properties"]["label"]["type"] == "string"
        assert submodel_node["properties"]["weight"]["type"] == "number"


class TestMakeStrictSchemaListField:
    """M7's generator response model (a list of proposed candidate windows)
    will almost certainly have a `list[SubModel]` field -- confirm
    `_make_strict_schema` handles that shape, both required and optional."""

    def test_list_items_ref_resolves_to_strict_object_in_defs(self) -> None:
        schema = _make_strict_schema(_ListFieldTestResponse.model_json_schema())

        windows_schema = schema["properties"]["windows"]
        assert windows_schema["type"] == "array"

        ref = windows_schema["items"]["$ref"]
        def_name = ref.rsplit("/", maxsplit=1)[-1]
        submodel_node = schema["$defs"][def_name]
        assert submodel_node["additionalProperties"] is False
        assert set(submodel_node["required"]) == set(submodel_node["properties"].keys())

    def test_optional_list_field_type_now_allows_null(self) -> None:
        schema = _make_strict_schema(_ListFieldTestResponse.model_json_schema())

        optional_windows_schema = schema["properties"]["optional_windows"]
        assert any(
            isinstance(branch, dict) and branch.get("type") == "null"
            for branch in optional_windows_schema["anyOf"]
        )
