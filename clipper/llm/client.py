"""core llm client (spec section 7): openai-compatible structured-output
calls, cached and retried, routed through the self-hosted freellmapi proxy
(see DECISIONS.md 2026-09-06 on freellmapi routing).

async, since spec section 4.7's judge stage fires up to 8 candidates
concurrently -- a later milestone builds that concurrency on top of
`complete_structured`, but the client itself has to be async from the start
to support it without a rewrite.

judge-specific fallback behavior ("invalid json or schema failure ...
after that, the candidate gets dimensions all 5, standalone_level 2, reason
'judge failed'", spec section 7) is deliberately NOT handled here: this
client raises `LLMSchemaError` once retries are exhausted, and the judge
stage (a later milestone) is responsible for catching it and applying that
judge-specific fallback. a generic client has no business knowing about
judge dimensions.
"""

from __future__ import annotations

import copy
import json
import os
import uuid
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from clipper.llm import cache
from clipper.llm.registry import resolve_model_alias
from clipper.util.log import append_event, record_stat

_MAX_ATTEMPTS = 3

# placeholder flat cost estimate -- freellmapi's real per-model billing /
# passthrough pricing is unknown until calibrated against its actual
# billing behavior. see DECISIONS.md.
_COST_PER_1K_TOKENS_USD = 0.0005


class LLMSchemaError(Exception):
    """raised when a structured-output call still fails json/schema
    validation after `_MAX_ATTEMPTS` attempts. carries the last validation
    error and the last raw response text for debugging."""

    def __init__(self, message: str, last_error: str, raw_response: str) -> None:
        super().__init__(message)
        self.last_error = last_error
        self.raw_response = raw_response


def _build_client() -> AsyncOpenAI:
    """`openai` sdk async client pointed at the self-hosted freellmapi
    proxy -- same `FREELLM_BASE_URL`/`FREELLM_API_KEY` env-var pattern as
    `clipper/transcribers/groq.py`'s `_build_client`. kept as a small
    duplicate rather than factored into a shared helper module: the two are
    only ~6 lines each, one sync/one async, and a new shared module isn't
    listed in spec section 6's repository layout -- not worth the extra
    indirection for this little duplication. see DECISIONS.md.
    """
    base_url = os.environ.get("FREELLM_BASE_URL")
    api_key = os.environ.get("FREELLM_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError(
            "FREELLM_BASE_URL and FREELLM_API_KEY must be set (see .env.example) "
            "-- every llm call routes through the freellmapi proxy"
        )
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def _json_schema_response_format(response_model: type[BaseModel]) -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": _make_strict_schema(response_model.model_json_schema()),
            "strict": True,
        },
    }


def _make_strict_schema(schema: dict) -> dict:
    """rewrite a pydantic `model_json_schema()` output to satisfy
    OpenAI-compatible strict structured-output validation, which
    `model_json_schema()` does not produce on its own: pydantic never emits
    `additionalProperties: false`, and it excludes optional/defaulted fields
    from `required` (see DECISIONS.md).

    walks every object-type node, including nested submodels under
    `$defs`, and for each one:
    - sets `additionalProperties: false`.
    - adds every property to `required` (strict mode requires ALL
      properties be required), and for any property pydantic did NOT
      originally mark required, adds `null` as an allowed type instead --
      the standard strict-mode workaround for optional/defaulted fields.
    """
    schema = copy.deepcopy(schema)
    defs = schema.get("$defs")
    if isinstance(defs, dict):
        schema["$defs"] = {name: _strict_schema_node(node) for name, node in defs.items()}
    return _strict_schema_node(schema)


def _strict_schema_node(node: dict) -> dict:
    """recursively apply strict-mode object rules to one schema node (and
    everything nested under it: object properties, array items, and
    `anyOf`/`oneOf`/`allOf` branches). see `_make_strict_schema`."""
    if not isinstance(node, dict):
        return node

    if isinstance(node.get("properties"), dict):
        original_required = set(node.get("required", []))
        node["additionalProperties"] = False
        new_properties = {}
        for prop_name, prop_schema in node["properties"].items():
            processed = _strict_schema_node(prop_schema)
            if prop_name not in original_required:
                processed = _allow_null_type(processed)
            new_properties[prop_name] = processed
        node["properties"] = new_properties
        node["required"] = list(node["properties"].keys())

    if isinstance(node.get("items"), dict):
        node["items"] = _strict_schema_node(node["items"])

    for combinator in ("anyOf", "oneOf", "allOf"):
        branches = node.get(combinator)
        if isinstance(branches, list):
            node[combinator] = [_strict_schema_node(branch) for branch in branches]

    return node


def _allow_null_type(schema_node: dict) -> dict:
    """add `null` as an allowed type/branch to `schema_node`, in place where
    possible. a bare `{"$ref": ...}` node can't take a sibling `type` key
    (json schema forbids it), so it's wrapped in an `anyOf` with a `null`
    branch instead -- the same shape pydantic/OpenAI use for `Optional`
    fields elsewhere in strict schemas."""
    if "$ref" in schema_node and len(schema_node) == 1:
        return {"anyOf": [schema_node, {"type": "null"}]}

    if "type" in schema_node:
        current_type = schema_node["type"]
        if isinstance(current_type, list):
            if "null" not in current_type:
                schema_node["type"] = [*current_type, "null"]
        else:
            schema_node["type"] = [current_type, "null"]
        return schema_node

    if isinstance(schema_node.get("anyOf"), list):
        if not any(
            isinstance(branch, dict) and branch.get("type") == "null"
            for branch in schema_node["anyOf"]
        ):
            schema_node["anyOf"] = [*schema_node["anyOf"], {"type": "null"}]
        return schema_node

    return {"anyOf": [schema_node, {"type": "null"}]}


def _estimate_cost_usd(tokens_in: int, tokens_out: int) -> float:
    """placeholder cost estimate: flat rate per 1k tokens (in + out priced
    the same). real figures need calibration against freellmapi's actual
    billing/passthrough behavior later. see DECISIONS.md."""
    return round((tokens_in + tokens_out) / 1000 * _COST_PER_1K_TOKENS_USD, 6)


def _record_call_stat(
    run_dir: Path,
    prompt_version: str,
    key: str,
    *,
    model: str,
    cache_hit: bool,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
) -> None:
    """record one llm call under `stats.json`'s `llm_calls` section, per
    spec section 7 ("record tokens in/out and estimated cost per call").
    `record_stat` overwrites a given `(section, key)` pair rather than
    accumulating, so each call gets its own unique key (a short uuid
    suffix) instead of colliding on `prompt_version` alone."""
    record_stat(
        run_dir,
        "llm_calls",
        f"{prompt_version}_{uuid.uuid4().hex[:12]}",
        {
            "model": model,
            "cache_hit": cache_hit,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd,
        },
    )


async def complete_structured[ResponseModelT: BaseModel](
    run_dir: Path,
    model_alias: str,
    prompt_version: str,
    system_prompt: str,
    user_prompt: str,
    response_model: type[ResponseModelT],
    temperature: float,
) -> ResponseModelT:
    """make one cached, structured-output llm call and return an instance
    of `response_model`.

    cache key = `sha256(model + prompt_version + input)` (spec section 7),
    where `input` is `system_prompt + user_prompt` -- everything that
    affects the model's output, deliberately excluding `temperature` (the
    spec's cache key formula is literally `model + prompt_version + input`,
    with no temperature component; see DECISIONS.md).

    on cache miss, retries up to `_MAX_ATTEMPTS` times total on invalid
    json / schema validation failure, appending the validation error to the
    next attempt's user prompt so the model can self-correct (spec section
    7). raises `LLMSchemaError` if every attempt fails.
    """
    model = resolve_model_alias(model_alias)
    key = cache.cache_key(model, prompt_version, system_prompt + user_prompt)

    append_event(
        run_dir,
        "llm",
        "call_started",
        {"model_alias": model_alias, "model": model, "prompt_version": prompt_version},
    )

    cached = cache.get_cached_response(run_dir, key)
    if cached is not None:
        append_event(run_dir, "llm", "cache_hit", {"model": model, "cache_key": key})
        _record_call_stat(
            run_dir,
            prompt_version,
            key,
            model=model,
            cache_hit=True,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
        )
        return response_model.model_validate(cached)

    append_event(run_dir, "llm", "cache_miss", {"model": model, "cache_key": key})

    client = _build_client()
    try:
        response_format = _json_schema_response_format(response_model)

        current_user_prompt = user_prompt
        last_error = ""
        last_raw = ""

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": current_user_prompt},
                ],
                response_format=response_format,
                temperature=temperature,
            )
            raw_content = completion.choices[0].message.content or ""
            last_raw = raw_content

            # every attempt is a real, billed api call -- whether it ends up
            # passing schema validation or not -- so its tokens/cost are
            # recorded here, unconditionally, once per attempt. recording only
            # the final successful attempt (as this used to) silently dropped
            # the cost of every failed attempt in a retry sequence, and dropped
            # ALL cost data when every attempt failed. see DECISIONS.md.
            usage = completion.usage
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else 0
            cost_usd = _estimate_cost_usd(tokens_in, tokens_out)
            _record_call_stat(
                run_dir,
                prompt_version,
                key,
                model=model,
                cache_hit=False,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
            )

            try:
                parsed = response_model.model_validate_json(raw_content)
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)
                append_event(
                    run_dir,
                    "llm",
                    "retry",
                    {"model": model, "attempt": attempt, "error": last_error},
                )
                current_user_prompt = (
                    f"{user_prompt}\n\nyour previous response failed validation with this "
                    f"error -- respond again with valid json matching the schema:\n{last_error}"
                )
                continue

            cache.store_cached_response(run_dir, key, json.loads(raw_content))
            append_event(
                run_dir,
                "llm",
                "call_succeeded",
                {
                    "model": model,
                    "attempts": attempt,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                },
            )
            return parsed

        append_event(
            run_dir,
            "llm",
            "call_failed",
            {"model": model, "attempts": _MAX_ATTEMPTS, "error": last_error},
        )
        raise LLMSchemaError(
            f"llm call failed schema validation after {_MAX_ATTEMPTS} attempts: {last_error}",
            last_error=last_error,
            raw_response=last_raw,
        )
    finally:
        # an unclosed async client leaves sockets for asyncio.run() to tear down
        # after the loop is gone, which prints "event loop is closed" noise.
        close = getattr(client, "close", None)  # test fakes have no close()
        if close is not None:
            await close()
