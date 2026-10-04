"""`clipper eval` runner (spec section 11): runs discovery on every golden
video and reports the metrics from `clipper/eval/metrics.py`.

golden layout: `<golden_dir>/<name>/{transcript.json, annotations.json}`.
transcription is skipped (the cached transcript is the input), so an eval run
never pays for it twice. `recall_pool` is measured on the judged pool because
the prefilter overwrites `candidates.json` in place and the wider pre-prefilter
pool is not kept (see DECISIONS.md).
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from clipper.config import Config
from clipper.eval import metrics
from clipper.eval.metrics import Interval
from clipper.pipeline import STAGE_ORDER, run_stage, total_llm_cost_usd
from clipper.util.log import read_stats

_STAGES_AFTER_TRANSCRIBE = STAGE_ORDER[STAGE_ORDER.index("atomize") :]
_HOUR_MS = 3_600_000


def _intervals(items: list[dict]) -> list[Interval]:
    return [Interval(i["start_ms"], i["end_ms"]) for i in items]


def evaluate_video(golden_video: Path, work_dir: Path, cfg: Config) -> dict[str, object]:
    annotations = json.loads((golden_video / "annotations.json").read_text(encoding="utf-8"))
    transcript = json.loads((golden_video / "transcript.json").read_text(encoding="utf-8"))

    run_dir = work_dir / golden_video.name
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    shutil.copy(golden_video / "transcript.json", run_dir / "transcript.json")
    (run_dir / "input.json").write_text(
        json.dumps(
            {
                "source_path": str(golden_video / "video.mp4"),
                "duration_ms": transcript["duration_ms"],
                "sha256": "golden",
            }
        )
    )

    run_cfg = cfg.model_copy(update={"content_type": annotations.get("content_type", cfg.content_type)})
    for name in _STAGES_AFTER_TRANSCRIBE:
        run_stage(name, run_dir, run_cfg)

    manifest = json.loads((run_dir / "selected.json").read_text(encoding="utf-8"))
    selected = [Interval(c["segments"][0]["start_ms"], c["segments"][-1]["end_ms"]) for c in manifest["clips"]]
    pool = _intervals(json.loads((run_dir / "judged.json").read_text(encoding="utf-8")))
    deduped = json.loads((run_dir / "deduped.json").read_text(encoding="utf-8"))
    selected_ranges = {tuple(c["atom_range"]) for c in manifest["clips"]}
    selected_embeddings = [
        c["embedding"] for c in deduped if (c["atom_start"], c["atom_end"]) in selected_ranges
    ]

    tier1 = _intervals([m for m in annotations["moments"] if m.get("tier") == 1])
    hours = transcript["duration_ms"] / _HOUR_MS
    cost = total_llm_cost_usd(run_dir)
    wall_ms = read_stats(run_dir).get("stages", {})
    wall_s = sum(v.get("wall_ms", 0) for v in wall_ms.values() if isinstance(v, dict)) / 1000

    return {
        "video": golden_video.name,
        "hit_at_n": metrics.hit_rate(tier1, selected),
        "recall_pool": metrics.recall_pool(tier1, pool),
        "trap_rate": metrics.trap_rate(_intervals(annotations.get("traps", [])), selected),
        "boundary_mae_ms": metrics.boundary_mae_ms(tier1, selected),
        "dup_rate": metrics.dup_rate(selected_embeddings),
        "spread_ms": metrics.spread(selected),
        "cost_per_hour_usd": cost / hours if hours else None,
        "wall_per_hour_s": wall_s / hours if hours else None,
    }


def run_eval(golden_dir: Path, cfg: Config, config_name: str, out_dir: Path) -> Path:
    videos = sorted(p for p in golden_dir.iterdir() if (p / "annotations.json").exists())
    if not videos:
        raise FileNotFoundError(f"no golden videos (dirs with annotations.json) under {golden_dir}")

    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    results = [evaluate_video(v, out_dir / "runs", cfg) for v in videos]
    report = {"config": config_name, "timestamp": timestamp, "videos": results}
    report_path = out_dir / f"{timestamp}_{config_name}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def render_table(report: dict) -> str:
    columns = ["video", "hit_at_n", "recall_pool", "trap_rate", "boundary_mae_ms", "dup_rate"]
    lines = ["  ".join(f"{c:>16}" for c in columns)]
    for row in report["videos"]:
        cells = [
            "n/a" if row[c] is None else (f"{row[c]:.2f}" if isinstance(row[c], float) else str(row[c]))
            for c in columns
        ]
        lines.append("  ".join(f"{c:>16}" for c in cells))
    return "\n".join(lines)
