"""judge stage (spec section 4.7): `run_dir/candidates.json` (M8's
prefiltered pool) + `run_dir/atoms.json` -> one llm call per candidate, run
concurrently (semaphore-limited to 8 in flight) -> `run_dir/judged.json`.

for each candidate, builds previous context (atoms up to 60s before),
the candidate's own atoms, and following context (atoms up to 30s after),
asks the judge llm (`clipper/prompts/judge_v1.md`) to score it, and computes
`JudgedCandidate.score` (spec section 8) from the result.

this is the first stage to actually apply the judge-specific fallback spec
section 7 describes but `clipper/llm/client.py` (M6) deliberately deferred:
on `LLMSchemaError` (retries exhausted) -- and, for the same "never crash"
reason `clipper/generators/llm_pass.py` (M7) already applies to its own llm
call, `openai.OpenAIError` too -- the candidate gets a fallback `JudgeResult`
(dimensions all 5, standalone_level 2, reason "judge failed") rather than
taking down the whole judge stage. see DECISIONS.md.

`score-at-judge-time` cross-milestone contract (see DECISIONS.md): the
formula in spec section 8 needs `penalties.boundary`/`penalties.silence`
(M10, refine boundaries) and `penalties.redundancy` (M13, dedupe), none of
which exist yet at this stage. `compute_score` here always treats all three
as `0.0` -- this is a genuinely correct application of the formula given the
information available now, not an approximation. `RefinedCandidate.score`
(M10) and later M13 MUST recompute `score` once real penalty values exist;
they do not inherit a "final" score from this stage.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from openai import OpenAIError
from pydantic import BaseModel, field_validator

from clipper.config import Config, ScoringWeights
from clipper.llm.client import LLMSchemaError, complete_structured
from clipper.types import (
    JUDGE_DIMENSIONS,
    Atom,
    Candidate,
    JudgedCandidate,
    JudgeResult,
    _validate_dimensions,
    _validate_standalone_level,
)
from clipper.util.log import append_event, record_stat

_STAGE = "judge"
_ATOMS_JSON_FILENAME = "atoms.json"
_CANDIDATES_JSON_FILENAME = "candidates.json"
_JUDGED_JSON_FILENAME = "judged.json"
_PROMPT_VERSION = "judge_v1"
_TEMPERATURE = 0.0  # spec section 7: "temperature 0 for judge"
_PREVIOUS_CONTEXT_WINDOW_MS = 60_000
_FOLLOWING_CONTEXT_WINDOW_MS = 30_000
_MAX_CONCURRENT_JUDGE_CALLS = 8  # spec section 4.7: "run concurrently (default 8 in flight)"
_KEYWORD_BONUS_MAX = 5.0  # spec section 8: "keyword_bonus max 5"
_PENALTY_NAMES = ("boundary", "silence", "redundancy")
_FALLBACK_REASON = "judge failed"  # spec section 7, verbatim

_STANDALONE_ADJUSTMENT: dict[int, float] = {0: 0.0, 1: -4.0, 2: -12.0}  # 3 is short-circuited, see compute_score

_SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "judge_v1.md").read_text(encoding="utf-8")


class JudgeDimensions(BaseModel):
    """the seven judge dimensions as named fields. a bare `dict[str, int]`
    renders as an object with no `properties`, which strict structured output
    collapses to `{}` (seen live against the proxy: every judge call came back
    with empty dimensions). named fields give the model keys to fill in.
    one field per `JUDGE_DIMENSIONS` entry; the range check stays in
    `_validate_dimensions` since min/max in the schema is not portable."""

    hook: int
    value: int
    emotion: int
    curiosity: int
    novelty: int
    completeness: int
    payoff: int


class JudgeLLMResponse(BaseModel):
    """the judge llm's raw response shape -- everything `JudgeResult` (spec
    section 3) has EXCEPT `candidate_id`. defined here, next to the code
    that calls the llm, rather than reusing `clipper.types.JudgeResult`
    directly as `complete_structured`'s `response_model` -- the model has no
    way to know (or reason to guess) the candidate's own id, so asking it to
    invent/echo one back is pointless and adds a spurious failure mode.
    `candidate_id` is instead set by `_judge_one` from the `Candidate` this
    response belongs to. same shape-model-lives-with-the-caller precedent as
    `clipper/generators/llm_pass.py`'s `GeneratorProposal`/`GeneratorOutput`.
    see DECISIONS.md.

    reuses `clipper.types`'s `_validate_dimensions`/`_validate_standalone_level`
    (not part of that module's public surface, but the single source of
    truth its own docstring calls out) so an out-of-range dimension score or
    standalone_level fails THIS model's validation -- which
    `complete_structured` retries against -- rather than only surfacing once
    the final `JudgeResult` is constructed from an already "successful" call.
    """

    dimensions: JudgeDimensions
    standalone_level: int
    suggested_atom_start: int
    suggested_atom_end: int
    reason: str
    title: str
    missing_context: str | None

    @field_validator("dimensions")
    @classmethod
    def _check_dimensions(cls, value: JudgeDimensions) -> JudgeDimensions:
        _validate_dimensions(value.model_dump())
        return value

    @field_validator("standalone_level")
    @classmethod
    def _check_standalone_level(cls, value: int) -> int:
        return _validate_standalone_level(value)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_score(
    judge: JudgeResult,
    weights: ScoringWeights,
    keyword_bonus: float,
    penalties: dict[str, float] | None = None,
) -> float:
    """spec section 8's exact formula. `penalties` defaults to none (`0.0`
    each), which is the correct input at judge time; refine and dedupe pass
    the real boundary/silence/redundancy values to recompute (see module
    docstring's cross-milestone contract note).

    `standalone_level == 3` ("depends on prior context") maps to "reject"
    in the spec's table, not a numeric adjustment -- the actual hard-reject
    enforcement lives in the gate stage (M11, spec section 4.9), not here.
    at this stage, a standalone_level == 3 candidate's score is meaningless
    (it will be gated out regardless of the number), so this short-circuits
    to exactly `0.0` without evaluating the base formula at all -- simpler
    and safer than inventing an undefined numeric substitute for "reject".
    see DECISIONS.md.
    """
    if judge.standalone_level == 3:
        return 0.0

    base = sum(getattr(weights, dimension) * judge.dimensions[dimension] for dimension in JUDGE_DIMENSIONS)
    standalone_adj = _STANDALONE_ADJUSTMENT[judge.standalone_level]
    penalty_total = sum((penalties or {}).get(name, 0.0) for name in _PENALTY_NAMES)
    raw_score = base * 10 + standalone_adj - penalty_total + keyword_bonus
    return _clamp(raw_score, 0.0, 100.0)


def compute_keyword_bonus(transcript_text: str, keywords: list[str]) -> float:
    """`5 * matched_keywords / len(keywords)`, `0` if no keywords are
    configured (spec section 8 gives the max but not the formula -- see
    DECISIONS.md for this milestone's interpretation). a keyword "matches"
    if it appears case-insensitively as a substring of the candidate's
    concatenated atom text."""
    if not keywords:
        return 0.0
    text_lower = transcript_text.lower()
    matched = sum(1 for keyword in keywords if keyword.lower() in text_lower)
    return _KEYWORD_BONUS_MAX * matched / len(keywords)


def _format_atom_line(atom: Atom) -> str:
    """`[id] speaker: text`, matching `clipper/generators/llm_pass.py`'s
    formatting convention. `speaker` is always empty in phase 1 (no
    diarization yet)."""
    speaker = atom.speaker or ""
    return f"[{atom.id}] {speaker}: {atom.text}"


def _render_atoms(atoms: list[Atom]) -> str:
    if not atoms:
        return "(none)"
    return "\n".join(_format_atom_line(atom) for atom in atoms)


def collect_previous_context(atoms_by_id: dict[int, Atom], atom_start: int, window_ms: int) -> list[Atom]:
    """atoms up to `window_ms` before the candidate's start (spec section
    4.7), walking backward from `atom_start - 1`. atom ids are contiguous
    0..n-1 (spec section 4.3), so this stops at the first missing id.

    boundary rule: an atom is included if its distance from the candidate's
    start (`candidate_start_ms - atom.start_ms`) is <= `window_ms`; the
    first (and therefore every earlier) atom whose distance exceeds
    `window_ms` is excluded, and the walk stops there. see DECISIONS.md.
    """
    start_atom = atoms_by_id.get(atom_start)
    if start_atom is None:
        return []

    candidate_start_ms = start_atom.start_ms
    context: list[Atom] = []
    atom_id = atom_start - 1
    while atom_id >= 0:
        atom = atoms_by_id.get(atom_id)
        if atom is None:
            break
        if candidate_start_ms - atom.start_ms > window_ms:
            break
        context.append(atom)
        atom_id -= 1

    context.reverse()
    return context


def collect_following_context(
    atoms_by_id: dict[int, Atom], atom_end: int, window_ms: int, max_atom_id: int
) -> list[Atom]:
    """atoms up to `window_ms` after the candidate's end (spec section 4.7),
    walking forward from `atom_end + 1`. same inclusive boundary rule as
    `collect_previous_context`, mirrored forward: an atom is included if its
    distance from the candidate's end (`atom.end_ms - candidate_end_ms`) is
    <= `window_ms`."""
    end_atom = atoms_by_id.get(atom_end)
    if end_atom is None:
        return []

    candidate_end_ms = end_atom.end_ms
    context: list[Atom] = []
    atom_id = atom_end + 1
    while atom_id <= max_atom_id:
        atom = atoms_by_id.get(atom_id)
        if atom is None:
            break
        if atom.end_ms - candidate_end_ms > window_ms:
            break
        context.append(atom)
        atom_id += 1

    return context


def _build_user_prompt(previous: list[Atom], candidate_atoms: list[Atom], following: list[Atom], cfg: Config) -> str:
    return (
        f"content_type: {cfg.content_type}\n"
        f"platform: {cfg.platform}\n"
        f"editorial_instructions: {cfg.editorial_instructions or '(none)'}\n"
        "\n"
        "previous_context:\n"
        f"{_render_atoms(previous)}\n"
        "\n"
        "candidate:\n"
        f"{_render_atoms(candidate_atoms)}\n"
        "\n"
        "following_context:\n"
        f"{_render_atoms(following)}\n"
    )


def _build_fallback_judge_result(candidate: Candidate) -> JudgeResult:
    """spec section 7's judge-specific fallback: "the candidate gets
    dimensions all 5, standalone_level 2, reason 'judge failed', and a log
    event. never crash." fields the spec doesn't specify a value for:

    - `title` (required on `JudgeResult`): falls back to the candidate's own
      `generator_hint` if one exists (still grounded in something real about
      the candidate), else the literal placeholder `"(judge failed)"`.
    - `suggested_atom_start`/`suggested_atom_end`: the candidate's own
      `atom_start`/`atom_end` -- i.e. no boundary adjustment recommended.
    - `missing_context`: `None`.

    see DECISIONS.md.
    """
    return JudgeResult(
        candidate_id=candidate.id,
        dimensions=dict.fromkeys(JUDGE_DIMENSIONS, 5),
        standalone_level=2,
        suggested_atom_start=candidate.atom_start,
        suggested_atom_end=candidate.atom_end,
        reason=_FALLBACK_REASON,
        title=candidate.generator_hint or "(judge failed)",
        missing_context=None,
    )


async def _judge_one(
    candidate: Candidate,
    atoms_by_id: dict[int, Atom],
    max_atom_id: int,
    cfg: Config,
    run_dir: Path,
    weights: ScoringWeights,
    semaphore: asyncio.Semaphore,
) -> tuple[JudgedCandidate, bool]:
    """judge one candidate, respecting `semaphore` (spec: "run concurrently,
    default 8 in flight"). returns the `JudgedCandidate` plus whether the
    judge-specific fallback (see `_build_fallback_judge_result`) was used."""
    async with semaphore:
        append_event(run_dir, _STAGE, "candidate_judge_started", {"candidate_id": candidate.id})

        previous_context = collect_previous_context(atoms_by_id, candidate.atom_start, _PREVIOUS_CONTEXT_WINDOW_MS)
        candidate_atoms = [atoms_by_id[i] for i in range(candidate.atom_start, candidate.atom_end + 1)]
        following_context = collect_following_context(
            atoms_by_id, candidate.atom_end, _FOLLOWING_CONTEXT_WINDOW_MS, max_atom_id
        )

        used_fallback = False
        try:
            response = await complete_structured(
                run_dir=run_dir,
                model_alias=cfg.llm_judge,
                prompt_version=_PROMPT_VERSION,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=_build_user_prompt(previous_context, candidate_atoms, following_context, cfg),
                response_model=JudgeLLMResponse,
                temperature=_TEMPERATURE,
            )
            judge_result = JudgeResult(candidate_id=candidate.id, **response.model_dump())
        except (LLMSchemaError, OpenAIError) as exc:
            # `LLMSchemaError` is spec section 7's explicit fallback trigger.
            # `openai.OpenAIError` (connection errors, timeouts, rate
            # limits, api status errors) is caught alongside it for the same
            # reason `clipper/generators/llm_pass.py` (M7) does: a network
            # blip is exactly as recoverable, from this stage's point of
            # view, as a schema failure -- and with up to 8 concurrent
            # candidates in flight, one candidate's transient network error
            # must not take down every other candidate's judge call. see
            # DECISIONS.md.
            used_fallback = True
            judge_result = _build_fallback_judge_result(candidate)
            append_event(
                run_dir,
                _STAGE,
                "judge_fallback_used",
                {"candidate_id": candidate.id, "error": str(exc)},
            )

        transcript_text = " ".join(atom.text for atom in candidate_atoms)
        keyword_bonus = compute_keyword_bonus(transcript_text, cfg.keywords)
        score = compute_score(judge_result, weights, keyword_bonus)

        judged_candidate = JudgedCandidate(**candidate.model_dump(), judge=judge_result, score=score)

        append_event(
            run_dir,
            _STAGE,
            "candidate_judge_completed",
            {"candidate_id": candidate.id, "score": score, "used_fallback": used_fallback},
        )

        return judged_candidate, used_fallback


async def _judge_all(
    candidates: list[Candidate],
    atoms_by_id: dict[int, Atom],
    max_atom_id: int,
    cfg: Config,
    run_dir: Path,
    weights: ScoringWeights,
) -> tuple[list[JudgedCandidate], int]:
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_JUDGE_CALLS)
    results = await asyncio.gather(
        *(
            _judge_one(candidate, atoms_by_id, max_atom_id, cfg, run_dir, weights, semaphore)
            for candidate in candidates
        )
    )
    judged = [judged_candidate for judged_candidate, _ in results]
    fallback_count = sum(1 for _, used_fallback in results if used_fallback)
    return judged, fallback_count


def run_judge(run_dir: Path, cfg: Config) -> None:
    """judge every candidate in `run_dir/candidates.json` (spec section
    4.7), writing `run_dir/judged.json`."""
    append_event(run_dir, _STAGE, "started", {})

    atoms = [
        Atom.model_validate(raw)
        for raw in json.loads((run_dir / _ATOMS_JSON_FILENAME).read_text(encoding="utf-8"))
    ]
    atoms_by_id = {atom.id: atom for atom in atoms}
    max_atom_id = max(atoms_by_id) if atoms_by_id else -1

    candidates = [
        Candidate.model_validate(raw)
        for raw in json.loads((run_dir / _CANDIDATES_JSON_FILENAME).read_text(encoding="utf-8"))
    ]

    weights = cfg.resolved_scoring_weights()

    wall_start = time.monotonic()
    judged, fallback_count = asyncio.run(_judge_all(candidates, atoms_by_id, max_atom_id, cfg, run_dir, weights))
    wall_ms = round((time.monotonic() - wall_start) * 1000)

    (run_dir / _JUDGED_JSON_FILENAME).write_text(
        json.dumps([judged_candidate.model_dump() for judged_candidate in judged], indent=2)
    )

    append_event(
        run_dir,
        _STAGE,
        "judged",
        {"count": len(judged), "fallback_count": fallback_count},
    )

    record_stat(run_dir, "candidates", "judged", len(judged))
    record_stat(run_dir, "judge", "fallback_count", fallback_count)
    record_stat(run_dir, "stages", _STAGE, {"wall_ms": wall_ms})
