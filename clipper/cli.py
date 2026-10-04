"""clipper command line (spec section 5). `render` and `eval` are later
phases and are deliberately not registered."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from clipper.config import Config
from clipper.eval.runner import render_table, run_eval
from clipper.pipeline import STAGE_ORDER, run_pipeline, run_stage
from clipper.report import render_cost, render_inspect
from clipper.util.timing import new_run_id

app = typer.Typer(no_args_is_help=True)

_RUNS_DIR = Path("runs")


def _check_stage(name: str) -> str:
    if name not in STAGE_ORDER:
        raise typer.BadParameter(f"unknown stage {name!r}; choose from: {', '.join(STAGE_ORDER)}")
    return name


def _load_config(run_dir: Path, config: Path | None) -> Config:
    """`--config` wins; otherwise reuse the config a prior stage resolved into
    input.json so `clipper stage` matches the original run."""
    if config is not None:
        return Config.from_yaml(config)
    input_path = run_dir / "input.json"
    if input_path.exists():
        return Config(**json.loads(input_path.read_text(encoding="utf-8"))["config"])
    return Config()


@app.callback()
def main() -> None:
    """clipper: long-form video to short-form clips."""
    load_dotenv()


@app.command()
def version() -> None:
    """print the installed clipper version."""
    print("clipper 0.1.0")


@app.command()
def run(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    config: Annotated[Path | None, typer.Option(help="yaml config file")] = None,
    out: Annotated[Path | None, typer.Option(help="run directory (default runs/<id>/)")] = None,
    stop_after: Annotated[str | None, typer.Option(help="stop after this stage")] = None,
) -> None:
    """run the full pipeline on a video."""
    if stop_after is not None:
        _check_stage(stop_after)
    cfg = Config.from_yaml(config)
    run_dir = out or _RUNS_DIR / new_run_id(video.name)
    run_pipeline(video, run_dir, cfg, stop_after)
    typer.echo(f"run directory: {run_dir}")
    if stop_after is None:
        typer.echo((run_dir / "selected.md").read_text(encoding="utf-8"))


@app.command()
def stage(
    name: Annotated[str, typer.Argument(help=f"one of: {', '.join(STAGE_ORDER)}")],
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    config: Annotated[Path | None, typer.Option(help="yaml config file")] = None,
) -> None:
    """re-run a single stage from its inputs in an existing run directory."""
    run_stage(_check_stage(name), run_dir, _load_config(run_dir, config))


@app.command()
def inspect(run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    """show per-stage counts and every candidate's fate."""
    typer.echo(render_inspect(run_dir))


@app.command()
def eval(
    golden_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    config: Annotated[Path | None, typer.Option(help="yaml config file")] = None,
    out: Annotated[Path, typer.Option(help="where eval/<timestamp>_<config>.json goes")] = Path("eval"),
) -> None:
    """run discovery on every golden video and report hit@n, recall_pool and friends."""
    report_path = run_eval(golden_dir, Config.from_yaml(config), config.stem if config else "default", out)
    typer.echo(render_table(json.loads(report_path.read_text(encoding="utf-8"))))
    typer.echo(f"saved: {report_path}")


@app.command()
def cost(run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    """sum the llm spend recorded for a run."""
    typer.echo(render_cost(run_dir))


if __name__ == "__main__":
    app()
