"""core pydantic v2 data types shared across every stage (spec section 3).

everything written to disk in a run directory is one of these serialized with
`model_dump_json()`. all timestamps are integer milliseconds; field names end
in `_ms`. the llm never sees or emits raw timestamps, only atom ids.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator

# fixed set of judge dimensions (spec section 3, 8). order matters for scoring
# but not for the dict itself; kept as a tuple so both the validator and any
# future scoring code share one source of truth.
JUDGE_DIMENSIONS: tuple[str, ...] = (
    "hook",
    "value",
    "emotion",
    "curiosity",
    "novelty",
    "completeness",
    "payoff",
)


def _validate_dimensions(value: dict[str, int]) -> dict[str, int]:
    """shared by `JudgeResult.dimensions` and `Clip.dimensions`: exactly the
    keys in JUDGE_DIMENSIONS, each an integer in [1, 10]."""
    expected = set(JUDGE_DIMENSIONS)
    actual = set(value.keys())

    missing = expected - actual
    if missing:
        raise ValueError(f"dimensions is missing required keys: {sorted(missing)}")

    extra = actual - expected
    if extra:
        raise ValueError(f"dimensions has unexpected extra keys: {sorted(extra)}")

    for key in JUDGE_DIMENSIONS:
        score = value[key]
        if not (1 <= score <= 10):
            raise ValueError(f"dimensions['{key}'] must be an integer in [1, 10], got {score}")

    return value


def _validate_standalone_level(value: int) -> int:
    """shared by `JudgeResult.standalone_level` and `Clip.standalone_level`:
    must be in [0, 3]."""
    if not (0 <= value <= 3):
        raise ValueError(f"standalone_level must be in [0, 3], got {value}")
    return value


class Word(BaseModel):
    text: str
    start_ms: int
    end_ms: int
    confidence: float | None = None


class Atom(BaseModel):
    """a sentence-ish unit. the smallest thing the llm may reference."""

    id: int  # 0-based, contiguous, stable within a run
    start_ms: int
    end_ms: int
    text: str
    words: list[Word]
    speaker: str | None = None
    pause_before_ms: int  # silence gap to previous atom, 0 for first
    pause_after_ms: int


class Transcript(BaseModel):
    language: str
    duration_ms: int
    words: list[Word]
    segments: list[dict]  # raw provider segments, kept for debugging
    provider: str
    model: str


class Candidate(BaseModel):
    id: str
    atom_start: int  # inclusive
    atom_end: int  # inclusive
    start_ms: int
    end_ms: int
    generators: list[str]  # which generators proposed this (merged)
    generator_hint: str | None  # free text from llm_pass, e.g. "a story about hiring"
    pre_score: float  # 0-1, deterministic


class JudgeResult(BaseModel):
    candidate_id: str
    dimensions: dict[str, int]  # keys fixed: see JUDGE_DIMENSIONS, each 1-10
    standalone_level: int  # 0 fully standalone, 1 minor gap, 2 imperfect, 3 depends on prior context
    suggested_atom_start: int
    suggested_atom_end: int
    reason: str  # <= 40 words
    title: str  # <= 12 words, must be grounded in transcript
    missing_context: str | None  # what the viewer would not understand

    @field_validator("dimensions")
    @classmethod
    def _check_dimensions(cls, value: dict[str, int]) -> dict[str, int]:
        return _validate_dimensions(value)

    @field_validator("standalone_level")
    @classmethod
    def _check_standalone_level(cls, value: int) -> int:
        return _validate_standalone_level(value)


class JudgedCandidate(Candidate):
    judge: JudgeResult
    score: float  # 0-100 computed by code, see spec section 8


class Segment(BaseModel):
    """a contiguous time range within a clip. `Clip.segments` is always a list;
    one element in the contiguous case (spec section 2.3)."""

    start_ms: int
    end_ms: int


class RefinedCandidate(JudgedCandidate):
    segments: list[Segment]
    refinement_log: list[str]
    penalties: dict[str, float]
    embedding: list[float] | None


class SourceInfo(BaseModel):
    """the manifest's "source" field (spec section 2.3)."""

    path: str
    duration_ms: int
    sha256: str


class Clip(BaseModel):
    """final selected clip as it appears in the manifest (spec section 2.3)."""

    id: str
    segments: list[Segment]
    duration_ms: int
    # note: modeled as a 2-tuple rather than a list[int] so the length is
    # enforced by the type system, matching the [start_atom, end_atom]
    # shape shown in the spec's example json (tuples serialize to json arrays).
    atom_range: tuple[int, int]
    score: float
    dimensions: dict[str, int]
    standalone_level: int
    penalties: dict[str, float]
    reason: str
    title: str
    transcript: str
    generators: list[str]
    speaker: str | None

    @field_validator("dimensions")
    @classmethod
    def _check_dimensions(cls, value: dict[str, int]) -> dict[str, int]:
        return _validate_dimensions(value)

    @field_validator("standalone_level")
    @classmethod
    def _check_standalone_level(cls, value: int) -> int:
        return _validate_standalone_level(value)


class ClipManifest(BaseModel):
    run_id: str
    source: SourceInfo
    config: dict
    clips: list[Clip]
    stats: dict
