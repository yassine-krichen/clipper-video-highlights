"""end-to-end run of every stage on the real fixture video.

marked `slow`: real ffmpeg/ffprobe and real local faster-whisper inference
(spec section 13: never mock local compute). only the network is replaced:
the llm calls get a deterministic offline stand-in and the embedder a
bag-of-words hasher, so the test needs no credentials or model download.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from clipper.config import Config
from clipper.generators import llm_pass
from clipper.pipeline import run_pipeline
from clipper.stages import dedupe, judge
from clipper.types import JUDGE_DIMENSIONS, ClipManifest

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_3min.mp4"
_ATOM_ID = re.compile(r"^\[(\d+)\]", re.MULTILINE)


async def _fake_complete_structured(*, user_prompt: str, response_model: type, **_: object):
    if response_model is llm_pass.GeneratorOutput:
        ids = [int(i) for i in _ATOM_ID.findall(user_prompt)]
        step = max(1, len(ids) // 4)
        proposals = [
            llm_pass.GeneratorProposal(
                atom_start=ids[i], atom_end=min(ids[-1], ids[i] + 6), hint=f"proposal {i}"
            )
            for i in range(0, len(ids), step)
        ]
        return llm_pass.GeneratorOutput(proposals=proposals)

    candidate_block = user_prompt.split("candidate:\n", 1)[1].split("\n\nfollowing_context:", 1)[0]
    ids = [int(i) for i in _ATOM_ID.findall(candidate_block)]
    # stable pseudo-judgement derived from the text so scores differ across candidates
    digest = hashlib.sha256(candidate_block.encode()).digest()
    return judge.JudgeLLMResponse(
        dimensions={name: 6 + digest[i] % 4 for i, name in enumerate(JUDGE_DIMENSIONS)},
        standalone_level=digest[8] % 2,
        suggested_atom_start=ids[0],
        suggested_atom_end=ids[-1],
        reason="offline stand-in judge",
        title=" ".join(candidate_block.split()[1:6]),
        missing_context=None,
    )


class _HashingEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * 64
            for word in text.lower().split():
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
            norm = sum(x * x for x in vector) ** 0.5 or 1.0
            vectors.append([x / norm for x in vector])
        return vectors


@pytest.mark.slow
def test_full_pipeline_produces_a_valid_manifest_and_inspectable_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm_pass, "complete_structured", _fake_complete_structured)
    monkeypatch.setattr(judge, "complete_structured", _fake_complete_structured)
    monkeypatch.setattr(dedupe, "_select_embedder", lambda cfg: _HashingEmbedder())

    run_dir = tmp_path / "run"
    cfg = Config(transcriber="faster_whisper", clip_count=3)

    run_pipeline(_FIXTURE, run_dir, cfg)

    manifest = ClipManifest.model_validate_json((run_dir / "selected.json").read_text())
    assert 1 <= len(manifest.clips) <= 3
    for clip in manifest.clips:
        assert cfg.min_duration_ms <= clip.duration_ms <= cfg.max_duration_ms
        assert clip.segments[0].end_ms <= manifest.source.duration_ms
        assert clip.transcript
    assert (run_dir / "selected.md").exists()

    stats = json.loads((run_dir / "stats.json").read_text())
    assert stats["candidates"]["selected"] == len(manifest.clips)
    assert "total_cost_usd" in stats["totals"]

    from clipper.report import render_inspect

    report = render_inspect(run_dir)
    assert "selected as clip_01" in report


def test_eval_reports_metrics_for_a_golden_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from clipper.eval.runner import run_eval

    monkeypatch.setattr(llm_pass, "complete_structured", _fake_complete_structured)
    monkeypatch.setattr(judge, "complete_structured", _fake_complete_structured)
    monkeypatch.setattr(dedupe, "_select_embedder", lambda cfg: _HashingEmbedder())
    golden = Path(__file__).parent.parent / "fixtures" / "golden"

    report_path = run_eval(golden, Config(clip_count=3), "test", tmp_path)

    report = json.loads(report_path.read_text())
    row = report["videos"][0]
    assert row["video"] == "sample_3min"
    assert row["recall_pool"] in (0.0, 1.0)
    assert row["trap_rate"] in (0.0, 1.0)
    assert row["cost_per_hour_usd"] == 0.0
