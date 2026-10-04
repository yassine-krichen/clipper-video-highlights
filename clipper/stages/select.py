"""select stage (spec section 4.11): `deduped.json` -> `selected.json`
(a `ClipManifest`) and `selected.md`.

greedy diversity-aware pick, `clip_count` rounds:
    score - 25 * max_cosine(already_selected) - 10 * timeline_proximity
"""

from __future__ import annotations

import json
from pathlib import Path

from clipper.config import Config
from clipper.embed.base import cosine
from clipper.stages.dedupe import transcript_text
from clipper.types import Atom, Clip, ClipManifest, RefinedCandidate, SourceInfo
from clipper.util.log import append_event, read_stats, record_stat
from clipper.util.timing import format_ms_as_mmss

_STAGE = "select"
_ATOMS_JSON_FILENAME = "atoms.json"
_DEDUPED_JSON_FILENAME = "deduped.json"
_SELECTED_JSON_FILENAME = "selected.json"
_SELECTED_MD_FILENAME = "selected.md"
_INPUT_JSON_FILENAME = "input.json"

_SIMILARITY_WEIGHT = 25.0
_PROXIMITY_WEIGHT = 10.0
_PROXIMITY_FULL_WITHIN_MS = 120_000
_PROXIMITY_ZERO_AT_MS = 600_000


def timeline_proximity(candidate: RefinedCandidate, picked: list[RefinedCandidate]) -> float:
    """1.0 if the nearest pick is within 120s, decaying linearly to 0.0 at 10 min.
    distance is the gap between the two intervals (0 when they overlap)."""
    if not picked:
        return 0.0
    gap_ms = min(
        max(0, max(candidate.start_ms, p.start_ms) - min(candidate.end_ms, p.end_ms))
        for p in picked
    )
    if gap_ms <= _PROXIMITY_FULL_WITHIN_MS:
        return 1.0
    if gap_ms >= _PROXIMITY_ZERO_AT_MS:
        return 0.0
    return 1.0 - (gap_ms - _PROXIMITY_FULL_WITHIN_MS) / (_PROXIMITY_ZERO_AT_MS - _PROXIMITY_FULL_WITHIN_MS)


def _max_cosine(candidate: RefinedCandidate, picked: list[RefinedCandidate]) -> float:
    similarities = [
        cosine(candidate.embedding, p.embedding)
        for p in picked
        if candidate.embedding and p.embedding
    ]
    return max(similarities, default=0.0)


def select_clips(candidates: list[RefinedCandidate], clip_count: int) -> list[RefinedCandidate]:
    remaining = list(candidates)
    picked: list[RefinedCandidate] = []
    while remaining and len(picked) < clip_count:
        best = max(
            remaining,
            key=lambda c: c.score
            - _SIMILARITY_WEIGHT * _max_cosine(c, picked)
            - _PROXIMITY_WEIGHT * timeline_proximity(c, picked),
        )
        picked.append(best)
        remaining.remove(best)
    return picked


def _to_clip(index: int, candidate: RefinedCandidate, atoms_by_id: dict[int, Atom]) -> Clip:
    return Clip(
        id=f"clip_{index:02d}",
        segments=candidate.segments,
        duration_ms=sum(s.end_ms - s.start_ms for s in candidate.segments),
        atom_range=(candidate.atom_start, candidate.atom_end),
        score=round(candidate.score, 1),
        dimensions=candidate.judge.dimensions,
        standalone_level=candidate.judge.standalone_level,
        penalties=candidate.penalties,
        reason=candidate.judge.reason,
        title=candidate.judge.title,
        transcript=transcript_text(candidate, atoms_by_id),
        generators=candidate.generators,
        speaker=atoms_by_id[candidate.atom_start].speaker,
    )


def _render_markdown(manifest: ClipManifest) -> str:
    lines = [f"# clips for {manifest.source.path}", ""]
    for clip in manifest.clips:
        start = format_ms_as_mmss(clip.segments[0].start_ms)
        end = format_ms_as_mmss(clip.segments[-1].end_ms)
        lines += [
            f"## {clip.id}: {clip.title}",
            f"- time: {start} - {end} ({clip.duration_ms / 1000:.1f}s)",
            f"- score: {clip.score}",
            f"- reason: {clip.reason}",
            "",
            clip.transcript,
            "",
        ]
    return "\n".join(lines)


def run_select(run_dir: Path, cfg: Config) -> None:
    append_event(run_dir, _STAGE, "started", {})

    atoms_by_id = {
        raw["id"]: Atom.model_validate(raw)
        for raw in json.loads((run_dir / _ATOMS_JSON_FILENAME).read_text(encoding="utf-8"))
    }
    deduped = [
        RefinedCandidate.model_validate(raw)
        for raw in json.loads((run_dir / _DEDUPED_JSON_FILENAME).read_text(encoding="utf-8"))
    ]
    picked = select_clips(deduped, cfg.clip_count)
    clips = [_to_clip(i, c, atoms_by_id) for i, c in enumerate(picked, start=1)]

    record_stat(run_dir, "candidates", "selected", len(clips))

    input_data = json.loads((run_dir / _INPUT_JSON_FILENAME).read_text(encoding="utf-8"))
    manifest = ClipManifest(
        run_id=run_dir.name,
        source=SourceInfo(
            path=Path(input_data["source_path"]).name,
            duration_ms=input_data["duration_ms"],
            sha256=input_data["sha256"],
        ),
        config=cfg.model_dump(),
        clips=clips,
        stats=read_stats(run_dir),
    )
    (run_dir / _SELECTED_JSON_FILENAME).write_text(manifest.model_dump_json(indent=2))
    (run_dir / _SELECTED_MD_FILENAME).write_text(_render_markdown(manifest), encoding="utf-8")
    append_event(run_dir, _STAGE, "selected", {"count": len(clips)})
