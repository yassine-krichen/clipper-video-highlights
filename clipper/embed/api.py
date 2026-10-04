"""api embedder (spec section 4.10, `embedder: api`): any openai-compatible
embeddings endpoint, here the same freellmapi proxy every other call uses."""

from __future__ import annotations

import os

from openai import OpenAI

from clipper.embed.base import normalize

# model name is a placeholder until verified against the proxy's real catalog
# (same caveat as clipper/llm/registry.py); overridable without a code change.
_DEFAULT_MODEL = "text-embedding-3-small"


class ApiEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        base_url = os.environ.get("FREELLM_BASE_URL")
        api_key = os.environ.get("FREELLM_API_KEY")
        if not base_url or not api_key:
            raise RuntimeError("FREELLM_BASE_URL and FREELLM_API_KEY must be set (see .env.example)")
        client = OpenAI(base_url=base_url, api_key=api_key)
        model = os.environ.get("CLIPPER_EMBEDDING_MODEL", _DEFAULT_MODEL)
        response = client.embeddings.create(model=model, input=texts)
        return [normalize(list(item.embedding)) for item in response.data]
