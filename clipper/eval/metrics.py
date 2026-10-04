"""evaluation metrics (spec section 11.2). pure functions over time intervals
and clip manifests; no i/o, so they are cheap to unit test."""

from __future__ import annotations

import math
from dataclasses import dataclass

from clipper.embed.base import cosine

_IOU_HIT_THRESHOLD = 0.5
_DUP_COSINE_THRESHOLD = 0.85


@dataclass(frozen=True)
class Interval:
    start_ms: int
    end_ms: int


def iou(a: Interval, b: Interval) -> float:
    overlap = min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms)
    if overlap <= 0:
        return 0.0
    union = (a.end_ms - a.start_ms) + (b.end_ms - b.start_ms) - overlap
    return overlap / union


def _best_match(target: Interval, pool: list[Interval]) -> Interval | None:
    scored = [(iou(target, p), p) for p in pool]
    scored = [pair for pair in scored if pair[0] >= _IOU_HIT_THRESHOLD]
    return max(scored, key=lambda pair: pair[0])[1] if scored else None


def hit_rate(moments: list[Interval], selected: list[Interval]) -> float | None:
    """hit@n: fraction of moments matched (iou >= 0.5) by a selected clip.
    `None` when there are no moments, so an empty annotation is not read as 0."""
    if not moments:
        return None
    return sum(_best_match(m, selected) is not None for m in moments) / len(moments)


def recall_pool(moments: list[Interval], pool: list[Interval]) -> float | None:
    """same measure as `hit_rate`, against the candidate pool instead."""
    return hit_rate(moments, pool)


def trap_rate(traps: list[Interval], selected: list[Interval]) -> float | None:
    """fraction of traps that got selected (iou >= 0.5)."""
    return hit_rate(traps, selected)


def boundary_mae_ms(moments: list[Interval], selected: list[Interval]) -> float | None:
    """mean absolute error of start and end versus the annotation, hits only."""
    errors: list[int] = []
    for moment in moments:
        match = _best_match(moment, selected)
        if match is not None:
            errors += [abs(match.start_ms - moment.start_ms), abs(match.end_ms - moment.end_ms)]
    return sum(errors) / len(errors) if errors else None


def dup_rate(embeddings: list[list[float]]) -> int:
    """number of selected clip pairs whose cosine similarity exceeds 0.85."""
    return sum(
        cosine(embeddings[i], embeddings[j]) > _DUP_COSINE_THRESHOLD
        for i in range(len(embeddings))
        for j in range(i + 1, len(embeddings))
    )


def spread(selected: list[Interval]) -> float | None:
    """std dev of selected clip midpoints, in ms. needs at least two clips."""
    if len(selected) < 2:
        return None
    midpoints = [(c.start_ms + c.end_ms) / 2 for c in selected]
    mean = sum(midpoints) / len(midpoints)
    return math.sqrt(sum((m - mean) ** 2 for m in midpoints) / len(midpoints))
