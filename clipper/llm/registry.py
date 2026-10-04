"""model alias registry (spec section 7).

spec section 7's original design assumed each alias maps to a different
llm *provider* (gemini, groq, claude, ...), each with its own base_url/api
key. per DECISIONS.md (2026-09-06), every llm call in this codebase instead
routes through one self-hosted openai-compatible proxy (freellmapi), which
routes internally by `model=` name. so this registry's job collapses to:
alias -> the model-name string to send to freellmapi.

CHECKED 2026-10-04 with a live chat call per name (the proxy's `GET /models`
catalog is NOT reliable: it listed gemini-2.5-flash and claude-sonnet-4-5 as
available, but the first rate-limited/404ed). gemini-flash, gemini-pro and
groq-llama answered; claude-sonnet did not (404, not in the catalog). the
free proxy's upstreams churn, so re-check an alias when judge calls start
falling back ("judge_fallback_used" events in log.jsonl). `gemini-pro` has no
real pro model behind it right now.

the model-name strings were originally PLACEHOLDERS. they follow each
provider's own public naming convention (e.g. google's "gemini-2.0-flash",
groq's "llama-3.3-70b-versatile") but this codebase has no way to verify
freellmapi's actual model catalog without calling the real, live proxy,
which the test suite must never do (spec section 13: mock only network
calls). verify/adjust these against freellmapi's real catalog before
depending on them for anything beyond development against mocks. see
DECISIONS.md.
"""

from __future__ import annotations

_MODEL_ALIASES: dict[str, str] = {
    "gemini-flash": "gemini-3.1-flash-lite",
    "gemini-pro": "gemini-3-flash-preview",
    "groq-llama": "llama-3.3-70b-fp8-fast",
    "claude-sonnet": "claude-sonnet-4-5",
}


class UnknownModelAliasError(Exception):
    """raised by `resolve_model_alias` for an alias not in `_MODEL_ALIASES`."""


def resolve_model_alias(alias: str) -> str:
    """resolve a config-facing model alias (e.g. `"gemini-flash"`) to the
    model-name string sent to freellmapi's `model=` parameter.

    raises `UnknownModelAliasError` for any alias not in the registry, with
    the list of known aliases in the message.
    """
    try:
        return _MODEL_ALIASES[alias]
    except KeyError:
        known = ", ".join(sorted(_MODEL_ALIASES))
        raise UnknownModelAliasError(
            f"unknown model alias {alias!r}; known aliases: {known}"
        ) from None
