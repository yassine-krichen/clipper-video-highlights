"""model alias registry (spec section 7).

spec section 7's original design assumed each alias maps to a different
llm *provider* (gemini, groq, claude, ...), each with its own base_url/api
key. per DECISIONS.md (2026-09-06), every llm call in this codebase instead
routes through one self-hosted openai-compatible proxy (freellmapi), which
routes internally by `model=` name. so this registry's job collapses to:
alias -> the model-name string to send to freellmapi.

IMPORTANT: the model-name strings below are PLACEHOLDERS. they follow each
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
    "gemini-flash": "gemini-2.0-flash",
    "gemini-pro": "gemini-2.5-pro",
    "groq-llama": "llama-3.3-70b-versatile",
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
