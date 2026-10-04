"""local embedder (spec section 4.10, `embedder: local`): sentence-transformers
`bge-small-en-v1.5`. the model is loaded lazily so importing this module (and
every run that never reaches dedupe) stays cheap."""

from __future__ import annotations

from clipper.embed.base import normalize

_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class LocalEmbedder:
    def __init__(self) -> None:
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(_MODEL_NAME)
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [normalize([float(x) for x in row]) for row in vectors]
