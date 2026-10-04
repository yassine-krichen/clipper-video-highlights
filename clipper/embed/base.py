"""embedder interface (spec section 4.10): dedupe and select only need
unit-length vectors, so every implementation returns L2-normalized rows and
cosine similarity reduces to a dot product."""

from __future__ import annotations

import math
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]


def cosine(a: list[float], b: list[float]) -> float:
    """dot product; equals cosine similarity because both inputs are unit length."""
    return sum(x * y for x, y in zip(a, b, strict=True))
