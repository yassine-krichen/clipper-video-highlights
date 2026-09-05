"""round-trip and validation tests for clipper/types.py (spec section 3)."""

import pytest
from pydantic import ValidationError

from clipper.types import (
    Atom,
    Candidate,
    Clip,
    ClipManifest,
    JudgedCandidate,
    JudgeResult,
    RefinedCandidate,
    Segment,
    SourceInfo,
    Transcript,
    Word,
)

VALID_DIMENSIONS = {
    "hook": 9,
    "value": 8,
    "emotion": 7,
    "curiosity": 9,
    "novelty": 8,
    "completeness": 9,
    "payoff": 8,
}


def _make_word() -> Word:
    return Word(text="hello", start_ms=0, end_ms=200, confidence=0.98)


def _make_atom() -> Atom:
    return Atom(
        id=0,
        start_ms=0,
        end_ms=1000,
        text="hello world",
        words=[_make_word(), _make_word()],
        speaker="spk_0",
        pause_before_ms=0,
        pause_after_ms=300,
    )


def _make_candidate() -> Candidate:
    return Candidate(
        id="cand_01",
        atom_start=412,
        atom_end=419,
        start_ms=2477200,
        end_ms=2512400,
        generators=["llm_pass", "discourse"],
        generator_hint="a story about hiring",
        pre_score=0.82,
    )


def _make_judge_result() -> JudgeResult:
    return JudgeResult(
        candidate_id="cand_01",
        dimensions=dict(VALID_DIMENSIONS),
        standalone_level=0,
        suggested_atom_start=412,
        suggested_atom_end=419,
        reason="opens on a concrete mistake, explains the cause, ends on the lesson.",
        title="the hiring mistake that cost us six months",
        missing_context=None,
    )


def _make_judged_candidate() -> JudgedCandidate:
    return JudgedCandidate(
        **_make_candidate().model_dump(),
        judge=_make_judge_result(),
        score=84.6,
    )


def _make_refined_candidate() -> RefinedCandidate:
    return RefinedCandidate(
        **_make_judged_candidate().model_dump(),
        segments=[Segment(start_ms=2477200, end_ms=2512400)],
        refinement_log=["shrunk trailing atom to satisfy max_duration_ms"],
        penalties={"boundary": 0.0, "silence": 1.2, "redundancy": 0.0},
        embedding=[0.1, 0.2, 0.3],
    )


def _make_clip() -> Clip:
    return Clip(
        id="clip_01",
        segments=[Segment(start_ms=2477200, end_ms=2512400)],
        duration_ms=35200,
        atom_range=(412, 419),
        score=84.6,
        dimensions=dict(VALID_DIMENSIONS),
        standalone_level=0,
        penalties={"boundary": 0.0, "silence": 1.2, "redundancy": 0.0},
        reason="opens on a concrete mistake, explains the cause, ends on the lesson.",
        title="the hiring mistake that cost us six months",
        transcript="the biggest mistake i made...",
        generators=["llm_pass", "discourse"],
        speaker=None,
    )


def _make_clip_manifest() -> ClipManifest:
    return ClipManifest(
        run_id="2026-09-05T14-02-11_ep17",
        source=SourceInfo(path="episode_17.mp4", duration_ms=3612400, sha256="a" * 64),
        config={"clip_count": 5},
        clips=[_make_clip()],
        stats={"total_cost_usd": 0.16},
    )


@pytest.mark.parametrize(
    "factory",
    [
        _make_word,
        _make_atom,
        lambda: Transcript(
            language="en",
            duration_ms=3612400,
            words=[_make_word()],
            segments=[{"id": 0, "text": "hello world"}],
            provider="groq",
            model="whisper-large-v3-turbo",
        ),
        _make_candidate,
        _make_judge_result,
        _make_judged_candidate,
        _make_refined_candidate,
        lambda: SourceInfo(path="episode_17.mp4", duration_ms=3612400, sha256="a" * 64),
        _make_clip,
        _make_clip_manifest,
    ],
)
def test_round_trip_serialization(factory: object) -> None:
    original = factory()  # type: ignore[operator]
    model_cls = type(original)
    round_tripped = model_cls.model_validate_json(original.model_dump_json())
    assert round_tripped == original


def test_judge_result_rejects_missing_dimension_key() -> None:
    dimensions = dict(VALID_DIMENSIONS)
    del dimensions["hook"]

    with pytest.raises(ValidationError, match="missing required keys"):
        JudgeResult(
            candidate_id="cand_01",
            dimensions=dimensions,
            standalone_level=0,
            suggested_atom_start=0,
            suggested_atom_end=1,
            reason="reason",
            title="title",
            missing_context=None,
        )


def test_judge_result_rejects_extra_dimension_key() -> None:
    dimensions = dict(VALID_DIMENSIONS)
    dimensions["virality"] = 5

    with pytest.raises(ValidationError, match="unexpected extra keys"):
        JudgeResult(
            candidate_id="cand_01",
            dimensions=dimensions,
            standalone_level=0,
            suggested_atom_start=0,
            suggested_atom_end=1,
            reason="reason",
            title="title",
            missing_context=None,
        )


def test_judge_result_rejects_dimension_value_below_range() -> None:
    dimensions = dict(VALID_DIMENSIONS)
    dimensions["hook"] = 0

    with pytest.raises(ValidationError, match=r"in \[1, 10\]"):
        JudgeResult(
            candidate_id="cand_01",
            dimensions=dimensions,
            standalone_level=0,
            suggested_atom_start=0,
            suggested_atom_end=1,
            reason="reason",
            title="title",
            missing_context=None,
        )


def test_judge_result_rejects_dimension_value_above_range() -> None:
    dimensions = dict(VALID_DIMENSIONS)
    dimensions["hook"] = 11

    with pytest.raises(ValidationError, match=r"in \[1, 10\]"):
        JudgeResult(
            candidate_id="cand_01",
            dimensions=dimensions,
            standalone_level=0,
            suggested_atom_start=0,
            suggested_atom_end=1,
            reason="reason",
            title="title",
            missing_context=None,
        )


@pytest.mark.parametrize("bad_level", [-1, 4])
def test_judge_result_rejects_out_of_range_standalone_level(bad_level: int) -> None:
    with pytest.raises(ValidationError, match="standalone_level must be in"):
        JudgeResult(
            candidate_id="cand_01",
            dimensions=dict(VALID_DIMENSIONS),
            standalone_level=bad_level,
            suggested_atom_start=0,
            suggested_atom_end=1,
            reason="reason",
            title="title",
            missing_context=None,
        )


def test_judge_result_accepts_standalone_level_upper_boundary() -> None:
    result = JudgeResult(
        candidate_id="cand_01",
        dimensions=dict(VALID_DIMENSIONS),
        standalone_level=3,
        suggested_atom_start=0,
        suggested_atom_end=1,
        reason="reason",
        title="title",
        missing_context=None,
    )
    assert result.standalone_level == 3


def test_judge_result_accepts_dimension_value_boundaries() -> None:
    dimensions = dict(VALID_DIMENSIONS)
    dimensions["hook"] = 1
    dimensions["value"] = 10

    result = JudgeResult(
        candidate_id="cand_01",
        dimensions=dimensions,
        standalone_level=0,
        suggested_atom_start=0,
        suggested_atom_end=1,
        reason="reason",
        title="title",
        missing_context=None,
    )
    assert result.dimensions["hook"] == 1
    assert result.dimensions["value"] == 10
