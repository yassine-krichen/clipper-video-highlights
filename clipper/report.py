"""`clipper inspect` / `clipper cost` rendering (spec section 10): every
candidate's fate, per-stage counts, and LLM spend, read from the run directory."""

from __future__ import annotations

import json
from pathlib import Path

from clipper.pipeline import total_llm_cost_usd
from clipper.util.log import read_stats


def _read_json(run_dir: Path, name: str) -> list[dict]:
    path = run_dir / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def candidate_fates(run_dir: Path) -> list[dict[str, object]]:
    """one row per candidate that reached the judge, with its last stage reached
    and, if rejected, why. candidates the prefilter dropped are not on disk."""
    rejected = {r["candidate_id"]: r for r in _read_json(run_dir, "rejected.json")}
    selected_path = run_dir / "selected.json"
    clip_by_range: dict[tuple[int, int], str] = {}
    if selected_path.exists():
        for clip in json.loads(selected_path.read_text(encoding="utf-8"))["clips"]:
            clip_by_range[tuple(clip["atom_range"])] = clip["id"]
    deduped_ids = {c["id"] for c in _read_json(run_dir, "deduped.json")}
    refined_by_id = {c["id"]: c for c in _read_json(run_dir, "refined.json")}

    rows: list[dict[str, object]] = []
    for judged in _read_json(run_dir, "judged.json"):
        cid = judged["id"]
        refined = refined_by_id.get(cid, judged)
        atom_range = (refined["atom_start"], refined["atom_end"])
        if cid in rejected:
            fate = f"rejected at {rejected[cid]['stage']}: {rejected[cid]['reason']}"
        elif atom_range in clip_by_range and cid in deduped_ids:
            fate = f"selected as {clip_by_range[atom_range]}"
        elif cid in deduped_ids:
            fate = "survived dedupe, not selected (clip_count reached)"
        else:
            fate = "stopped before the end of the pipeline"
        rows.append(
            {
                "id": cid,
                "atoms": f"{atom_range[0]}-{atom_range[1]}",
                "score": round(refined["score"], 1),
                "fate": fate,
            }
        )
    return sorted(rows, key=lambda r: r["score"], reverse=True)


def render_inspect(run_dir: Path) -> str:
    stats = read_stats(run_dir)
    lines = [f"run: {run_dir.name}", "", "candidate counts per stage:"]
    for key, value in stats.get("candidates", {}).items():
        lines.append(f"  {key}: {value}")

    lines += ["", "candidates (best score first):"]
    for row in candidate_fates(run_dir):
        lines.append(f"  {row['id']}  atoms {row['atoms']}  score {row['score']}  {row['fate']}")

    lines += ["", f"llm cost: ${total_llm_cost_usd(run_dir):.4f}"]
    return "\n".join(lines)


def render_cost(run_dir: Path) -> str:
    calls = read_stats(run_dir).get("llm_calls", {})
    billed = [c for c in calls.values() if not c.get("cache_hit")]
    tokens_in = sum(c.get("tokens_in", 0) for c in billed)
    tokens_out = sum(c.get("tokens_out", 0) for c in billed)
    return (
        f"llm calls: {len(calls)} ({len(calls) - len(billed)} cache hits)\n"
        f"tokens in/out: {tokens_in}/{tokens_out}\n"
        f"estimated cost: ${total_llm_cost_usd(run_dir):.4f}"
    )
