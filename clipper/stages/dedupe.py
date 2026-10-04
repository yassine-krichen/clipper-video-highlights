"""dedupe stage (spec section 4.10): `gated.json` -> `deduped.json`.

temporal pass (interval iou > 0.35), then semantic pass (embedding cosine
> 0.85). in both, candidates are visited best-score first and a later one that
overlaps an already kept one loses. a semantic loser's score is recomputed with
`penalties.redundancy` so it can be inspected; losers go to `rejected.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

from clipper.config import Config
from clipper.embed.api import ApiEmbedder
from clipper.embed.base import Embedder, cosine
from clipper.embed.local import LocalEmbedder
from clipper.stages.judge import compute_keyword_bonus, compute_score
from clipper.types import Atom, RefinedCandidate
from clipper.util.log import append_event, record_stat

_STAGE = "dedupe"
_ATOMS_JSON_FILENAME = "atoms.json"
_GATED_JSON_FILENAME = "gated.json"
_DEDUPED_JSON_FILENAME = "deduped.json"
_REJECTED_JSON_FILENAME = "rejected.json"

_TEMPORAL_IOU_THRESHOLD = 0.35
_SEMANTIC_COSINE_THRESHOLD = 0.85
# the penalty is informational for a rejected loser, scaled like select's diversity term (spec 4.11).
_REDUNDANCY_PENALTY_PER_COSINE = 25.0


def interval_iou(a: RefinedCandidate, b: RefinedCandidate) -> float:
    overlap = min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms)
    if overlap <= 0:
        return 0.0
    union = (a.end_ms - a.start_ms) + (b.end_ms - b.start_ms) - overlap
    return overlap / union


def _select_embedder(cfg: Config) -> Embedder:
    return ApiEmbedder() if cfg.embedder == "api" else LocalEmbedder()


def dedupe_temporal(
    candidates: list[RefinedCandidate],
) -> tuple[list[RefinedCandidate], list[dict[str, str]]]:
    kept: list[RefinedCandidate] = []
    rejected: list[dict[str, str]] = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        winner = next((k for k in kept if interval_iou(candidate, k) > _TEMPORAL_IOU_THRESHOLD), None)
        if winner is None:
            kept.append(candidate)
        else:
            rejected.append(
                {
                    "candidate_id": candidate.id,
                    "stage": _STAGE,
                    "reason": f"temporal overlap with {winner.id} (iou > {_TEMPORAL_IOU_THRESHOLD})",
                }
            )
    return kept, rejected


def dedupe_semantic(
    candidates: list[RefinedCandidate], cfg: Config, atoms_by_id: dict[int, Atom]
) -> tuple[list[RefinedCandidate], list[dict[str, str]]]:
    """expects `candidates` score-sorted with `embedding` already set."""
    kept: list[RefinedCandidate] = []
    rejected: list[dict[str, str]] = []
    for candidate in candidates:
        assert candidate.embedding is not None
        best = max(
            ((cosine(candidate.embedding, k.embedding), k) for k in kept if k.embedding),
            key=lambda pair: pair[0],
            default=None,
        )
        if best is None or best[0] <= _SEMANTIC_COSINE_THRESHOLD:
            kept.append(candidate)
            continue
        similarity, winner = best
        penalties = {**candidate.penalties, "redundancy": _REDUNDANCY_PENALTY_PER_COSINE * similarity}
        text = " ".join(atoms_by_id[i].text for i in range(candidate.atom_start, candidate.atom_end + 1))
        recomputed = compute_score(
            candidate.judge,
            cfg.resolved_scoring_weights(),
            compute_keyword_bonus(text, cfg.keywords),
            penalties,
        )
        rejected.append(
            {
                "candidate_id": candidate.id,
                "stage": _STAGE,
                "reason": (
                    f"semantic duplicate of {winner.id} (cosine {similarity:.2f}); "
                    f"score with redundancy penalty {recomputed:.1f}"
                ),
            }
        )
    return kept, rejected


def transcript_text(candidate: RefinedCandidate, atoms_by_id: dict[int, Atom]) -> str:
    return " ".join(atoms_by_id[i].text for i in range(candidate.atom_start, candidate.atom_end + 1))


def run_dedupe(run_dir: Path, cfg: Config, embedder: Embedder | None = None) -> None:
    append_event(run_dir, _STAGE, "started", {})

    atoms_by_id = {
        raw["id"]: Atom.model_validate(raw)
        for raw in json.loads((run_dir / _ATOMS_JSON_FILENAME).read_text(encoding="utf-8"))
    }
    gated = [
        RefinedCandidate.model_validate(raw)
        for raw in json.loads((run_dir / _GATED_JSON_FILENAME).read_text(encoding="utf-8"))
    ]

    temporal_kept, rejected = dedupe_temporal(gated)

    embedder = embedder or _select_embedder(cfg)
    vectors = embedder.embed([transcript_text(c, atoms_by_id) for c in temporal_kept])
    embedded = [
        c.model_copy(update={"embedding": vector})
        for c, vector in zip(temporal_kept, vectors, strict=True)
    ]
    kept, semantic_rejected = dedupe_semantic(embedded, cfg, atoms_by_id)
    rejected += semantic_rejected

    (run_dir / _DEDUPED_JSON_FILENAME).write_text(json.dumps([c.model_dump() for c in kept], indent=2))

    # gate wrote the earlier rejects to the same file; append rather than overwrite.
    rejected_path = run_dir / _REJECTED_JSON_FILENAME
    previous = json.loads(rejected_path.read_text(encoding="utf-8")) if rejected_path.exists() else []
    previous = [r for r in previous if r.get("stage") != _STAGE]
    rejected_path.write_text(json.dumps(previous + rejected, indent=2))

    for entry in rejected:
        append_event(run_dir, _STAGE, "rejected", entry)
    record_stat(run_dir, "candidates", "after_dedupe", len(kept))
