"""g1 `llm_pass` generator (spec section 4.5, primary generator): one llm
call over the whole transcript, asking for 20-40 candidate clip windows.

unlike g2-g4, this generator's signature is
`async def generate_llm_pass(atoms, cfg, run_dir) -> list[Candidate]`: it
needs `run_dir` for `complete_structured`'s (`clipper/llm/client.py`, M6)
cache/stats/event bookkeeping, and it's `async` because `complete_structured`
is. see `clipper/generators/base.py`'s module docstring for why this isn't
forced into the same `Protocol` as g2/g3.

the response model (`GeneratorProposal`/`GeneratorOutput`) is defined here,
not in `clipper/types.py`, per the phase-1 plan's design decision that
llm-response-shape models belong next to the code that calls the llm, not in
the shared on-disk data types module.
"""

from __future__ import annotations

from pathlib import Path

from openai import OpenAIError
from pydantic import BaseModel

from clipper.config import Config
from clipper.generators.base import build_candidate
from clipper.llm.client import LLMSchemaError, complete_structured
from clipper.types import Atom, Candidate
from clipper.util.log import append_event

_GENERATOR_NAME = "llm_pass"
_PROMPT_VERSION = "generator_v1"
_TEMPERATURE = 0.3  # spec section 7: "temperature 0 for judge, 0.3 for generator"

_SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "generator_v1.md").read_text(
    encoding="utf-8"
)

# spec: "if the transcript exceeds the model's practical limit (~150k
# tokens), split into 45-minute chunks with 5-minute overlap and merge."
# token count is estimated as chars/4 -- a commonly-cited rough
# approximation for English text, good enough for a threshold check. a real
# tokenizer dependency (e.g. tiktoken) would be overkill just to decide
# whether to chunk; this estimate is never used for cost accounting (that
# uses the api response's real token counts, see `clipper/llm/client.py`'s
# `_estimate_cost_usd`). see DECISIONS.md.
_CHARS_PER_TOKEN_ESTIMATE = 4.0
_TOKEN_LIMIT_FOR_CHUNKING = 150_000
_CHUNK_DURATION_MS = 45 * 60 * 1000
_CHUNK_OVERLAP_MS = 5 * 60 * 1000


class GeneratorProposal(BaseModel):
    atom_start: int
    atom_end: int
    hint: str


class GeneratorOutput(BaseModel):
    proposals: list[GeneratorProposal]


def _estimate_token_count(text: str) -> int:
    """rough token-count estimate (chars/4). see module-level comment above
    `_CHARS_PER_TOKEN_ESTIMATE`."""
    return int(len(text) / _CHARS_PER_TOKEN_ESTIMATE)


def _format_atom_line(atom: Atom) -> str:
    """`[id] speaker: text` per spec section 4.5. `speaker` renders as an
    empty string in phase 1, since `Atom.speaker` is always `None` (no
    diarization yet, see `clipper/stages/atomize.py`'s module docstring)."""
    speaker = atom.speaker or ""
    return f"[{atom.id}] {speaker}: {atom.text}"


def _build_user_prompt(atoms: list[Atom], cfg: Config) -> str:
    atom_lines = "\n".join(_format_atom_line(atom) for atom in atoms)
    keywords = ", ".join(cfg.keywords) if cfg.keywords else "(none)"
    editorial_instructions = cfg.editorial_instructions or "(none)"
    return (
        f"content_type: {cfg.content_type}\n"
        f"keywords: {keywords}\n"
        f"editorial_instructions: {editorial_instructions}\n"
        f"target clip duration: {cfg.min_duration_ms}ms to {cfg.max_duration_ms}ms\n"
        "\n"
        "transcript atoms:\n"
        f"{atom_lines}\n"
    )


def _chunk_atoms_by_time(atoms: list[Atom], chunk_duration_ms: int, overlap_ms: int) -> list[list[Atom]]:
    """split `atoms` into time-based chunks, each spanning up to
    `chunk_duration_ms`, with consecutive chunks overlapping by
    `overlap_ms` -- same spirit as `clipper/transcribers/groq.py`'s
    `chunk_wav`, but operating on atoms (by `start_ms`) rather than raw
    audio bytes. an atom belongs to chunk `i` if its `start_ms` falls in
    `[i * (chunk_duration_ms - overlap_ms), that + chunk_duration_ms)`.
    """
    if not atoms:
        return []

    total_span_ms = atoms[-1].end_ms
    step_ms = chunk_duration_ms - overlap_ms

    chunks: list[list[Atom]] = []
    chunk_start_ms = 0
    while chunk_start_ms < total_span_ms:
        chunk_end_ms = chunk_start_ms + chunk_duration_ms
        chunk_atoms = [atom for atom in atoms if chunk_start_ms <= atom.start_ms < chunk_end_ms]
        if chunk_atoms:
            chunks.append(chunk_atoms)
        chunk_start_ms += step_ms

    return chunks


async def generate_llm_pass(atoms: list[Atom], cfg: Config, run_dir: Path) -> list[Candidate]:
    """g1: see module docstring.

    merging across chunks is simpler than `clipper/transcribers/groq.py`'s
    audio stitching: atom ids are already global/absolute (assigned once by
    `clipper/stages/atomize.py` over the whole transcript), so there's no
    timing-based word-matching to do -- merging chunk proposals is just a
    union with de-duplication on `(atom_start, atom_end)`, since two
    overlapping chunks can both propose the same range.
    """
    if not atoms:
        return []

    full_user_prompt = _build_user_prompt(atoms, cfg)
    if _estimate_token_count(_SYSTEM_PROMPT + full_user_prompt) <= _TOKEN_LIMIT_FOR_CHUNKING:
        chunks = [atoms]
    else:
        chunks = _chunk_atoms_by_time(atoms, _CHUNK_DURATION_MS, _CHUNK_OVERLAP_MS)

    atoms_by_id = {atom.id: atom for atom in atoms}
    seen_ranges: set[tuple[int, int]] = set()
    candidates: list[Candidate] = []

    for chunk_atoms in chunks:
        try:
            output = await complete_structured(
                run_dir=run_dir,
                model_alias=cfg.llm_generator,
                prompt_version=_PROMPT_VERSION,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=_build_user_prompt(chunk_atoms, cfg),
                response_model=GeneratorOutput,
                temperature=_TEMPERATURE,
            )
        except (LLMSchemaError, OpenAIError) as exc:
            # `complete_structured` (M6) deliberately raises `LLMSchemaError`
            # after exhausting its retries rather than applying a fallback
            # itself -- that's left to the caller. the judge stage (a later
            # milestone) has a judge-specific fallback to apply; `llm_pass`
            # has none, so per spec section 1 ("invalid llm output never
            # crashes a run") the right degradation here is to contribute
            # zero candidates for this chunk rather than propagate and take
            # down the whole `generate` stage.
            #
            # `openai.OpenAIError` (the SDK's common base for connection
            # errors, timeouts, rate limits, and api status errors) is
            # caught alongside `LLMSchemaError` for the same reason: a
            # network blip on g1 is exactly as recoverable as a schema
            # failure from this generator's point of view -- g2-g4 are fully
            # deterministic and have nothing to do with g1's llm call, so
            # they shouldn't go down with it either.
            append_event(
                run_dir,
                "generate",
                "llm_pass_failed",
                {"error": str(exc), "chunk_atom_count": len(chunk_atoms)},
            )
            continue

        for proposal in output.proposals:
            range_key = (proposal.atom_start, proposal.atom_end)
            if range_key in seen_ranges:
                continue

            start_atom = atoms_by_id.get(proposal.atom_start)
            end_atom = atoms_by_id.get(proposal.atom_end)
            if start_atom is None or end_atom is None or proposal.atom_start > proposal.atom_end:
                # defensive: the model could hallucinate an atom id outside
                # the transcript it was given, or an inverted range. drop
                # rather than crash the whole generate stage over one bad
                # proposal -- the other 20-40 proposals are still useful.
                append_event(
                    run_dir,
                    "generate",
                    "llm_pass_proposal_dropped",
                    {
                        "atom_start": proposal.atom_start,
                        "atom_end": proposal.atom_end,
                        "reason": "unresolvable atom id or inverted range",
                    },
                )
                continue

            seen_ranges.add(range_key)
            candidates.append(build_candidate(start_atom, end_atom, _GENERATOR_NAME, hint=proposal.hint))

    return candidates
