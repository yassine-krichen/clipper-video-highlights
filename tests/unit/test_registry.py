"""tests for `clipper/llm/registry.py`'s model alias resolution."""

from __future__ import annotations

import pytest

from clipper.llm.registry import UnknownModelAliasError, resolve_model_alias


class TestResolveModelAlias:
    @pytest.mark.parametrize("alias", ["gemini-flash", "gemini-pro", "groq-llama", "claude-sonnet"])
    def test_resolves_known_aliases_to_nonempty_model_names(self, alias: str) -> None:
        model_name = resolve_model_alias(alias)

        assert isinstance(model_name, str)
        assert model_name

    def test_raises_clear_error_for_unknown_alias(self) -> None:
        with pytest.raises(UnknownModelAliasError, match="not-a-real-alias"):
            resolve_model_alias("not-a-real-alias")
