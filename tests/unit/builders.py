"""hand-built atoms and candidates shared by the refine/gate/dedupe/select tests."""

from __future__ import annotations

from clipper.types import (
    JUDGE_DIMENSIONS,
    Atom,
    JudgedCandidate,
    JudgeResult,
    RefinedCandidate,
    Segment,
    Word,
)


def make_atom(
    atom_id: int,
    start_ms: int,
    end_ms: int,
    text: str | None = None,
    *,
    pause_before_ms: int = 0,
    pause_after_ms: int = 0,
    confidence: float | None = None,
) -> Atom:
    text = text if text is not None else f"atom {atom_id}"
    return Atom(
        id=atom_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        words=[
            Word(text=w, start_ms=start_ms, end_ms=end_ms, confidence=confidence)
            for w in text.split()
        ],
        pause_before_ms=pause_before_ms,
        pause_after_ms=pause_after_ms,
    )


def make_judge(
    candidate_id: str,
    *,
    level: int = 0,
    dim: int = 8,
    suggested: tuple[int, int] = (0, 0),
) -> JudgeResult:
    return JudgeResult(
        candidate_id=candidate_id,
        dimensions=dict.fromkeys(JUDGE_DIMENSIONS, dim),
        standalone_level=level,
        suggested_atom_start=suggested[0],
        suggested_atom_end=suggested[1],
        reason="because",
        title="a title",
        missing_context=None,
    )


def make_judged(
    candidate_id: str,
    atom_start: int,
    atom_end: int,
    start_ms: int,
    end_ms: int,
    *,
    level: int = 0,
    dim: int = 8,
    suggested: tuple[int, int] | None = None,
) -> JudgedCandidate:
    return JudgedCandidate(
        id=candidate_id,
        atom_start=atom_start,
        atom_end=atom_end,
        start_ms=start_ms,
        end_ms=end_ms,
        generators=["llm_pass"],
        generator_hint=None,
        pre_score=0.5,
        judge=make_judge(
            candidate_id, level=level, dim=dim, suggested=suggested or (atom_start, atom_end)
        ),
        score=dim * 10.0,
    )


def make_refined(
    candidate_id: str,
    start_ms: int,
    end_ms: int,
    score: float,
    *,
    atom_start: int = 0,
    atom_end: int = 0,
    level: int = 0,
    embedding: list[float] | None = None,
) -> RefinedCandidate:
    judged = make_judged(candidate_id, atom_start, atom_end, start_ms, end_ms, level=level)
    fields = judged.model_dump()
    fields["score"] = score
    return RefinedCandidate(
        **fields,
        segments=[Segment(start_ms=start_ms, end_ms=end_ms)],
        refinement_log=[],
        penalties={"boundary": 0.0, "silence": 0.0, "redundancy": 0.0},
        embedding=embedding,
    )
