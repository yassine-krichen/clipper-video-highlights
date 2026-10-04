"""stage orchestration (spec section 4): the ordered stage table behind
`clipper run` and `clipper stage`."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from clipper.config import Config
from clipper.stages.atomize import run_atomize
from clipper.stages.dedupe import run_dedupe
from clipper.stages.gate import run_gate
from clipper.stages.generate import run_generate
from clipper.stages.judge import run_judge
from clipper.stages.prefilter import run_prefilter
from clipper.stages.probe import run_probe
from clipper.stages.refine import run_refine
from clipper.stages.select import run_select
from clipper.stages.transcribe import run_transcribe
from clipper.util.log import append_event, read_stats, record_stat

StageFn = Callable[[Path, Config], None]

# every stage after probe takes only (run_dir, cfg); probe also needs the source video.
STAGES: dict[str, StageFn] = {
    "transcribe": run_transcribe,
    "atomize": run_atomize,
    "generate": run_generate,
    "prefilter": run_prefilter,
    "judge": run_judge,
    "refine": run_refine,
    "gate": run_gate,
    "dedupe": run_dedupe,
    "select": run_select,
}
STAGE_ORDER: tuple[str, ...] = ("probe", *STAGES)


def total_llm_cost_usd(run_dir: Path) -> float:
    calls = read_stats(run_dir).get("llm_calls", {})
    return round(sum(call.get("cost_usd", 0.0) for call in calls.values()), 6)


def source_path_from_input(run_dir: Path) -> Path:
    input_path = run_dir / "input.json"
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found: run the probe stage with a video first")
    return Path(json.loads(input_path.read_text(encoding="utf-8"))["source_path"])


def run_stage(name: str, run_dir: Path, cfg: Config, video: Path | None = None) -> None:
    if name == "probe":
        run_probe(run_dir, cfg, video or source_path_from_input(run_dir))
        return
    STAGES[name](run_dir, cfg)


def run_pipeline(video: Path, run_dir: Path, cfg: Config, stop_after: str | None = None) -> None:
    """run stages in order, stopping after `stop_after` if given. wall time and
    LLM cost are recorded before select so the manifest's stats include them."""
    wall_start = time.monotonic()
    for name in STAGE_ORDER:
        if name == "select":
            record_stat(run_dir, "totals", "total_cost_usd", total_llm_cost_usd(run_dir))
            record_stat(run_dir, "totals", "total_wall_ms", round((time.monotonic() - wall_start) * 1000))
        append_event(run_dir, "pipeline", "stage_started", {"stage": name})
        run_stage(name, run_dir, cfg, video)
        if name == stop_after:
            break
