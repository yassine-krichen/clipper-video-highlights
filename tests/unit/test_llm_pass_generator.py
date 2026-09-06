"""tests for `clipper/generators/llm_pass.py` (g1, spec section 4.5).

per spec section 13, only the network call
(`AsyncOpenAI.chat.completions.create`) is mocked -- mirrors
`tests/unit/test_llm_client.py`'s approach, since this exercises the real
integration path through `complete_structured` (M6), including its
strict-schema handling for `GeneratorOutput`'s `list[GeneratorProposal]`
field.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipper.config import Config
from clipper.generators import llm_pass
from clipper.llm import cache as llm_cache
from clipper.types import Atom, Word
from clipper.util.log import read_events


def _atom(atom_id: int, start_ms: int, end_ms: int) -> Atom:
    return Atom(
        id=atom_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text=f"this is atom {atom_id}",
        words=[Word(text=f"atom{atom_id}", start_ms=start_ms, end_ms=end_ms)],
        pause_before_ms=0,
        pause_after_ms=0,
    )


@pytest.fixture(autouse=True)
def _freellm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FREELLM_BASE_URL", "https://freellmapi.test/v1")
    monkeypatch.setenv("FREELLM_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def _isolated_global_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_cache, "_GLOBAL_CACHE_DIR", tmp_path / "global_cache")


def _fake_completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _patch_client(monkeypatch: pytest.MonkeyPatch, fake_create) -> None:
    from clipper.llm import client as llm_client

    monkeypatch.setattr(
        llm_client,
        "_build_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        ),
    )


class TestGenerateLlmPassSingleCall:
    @pytest.mark.asyncio
    async def test_converts_proposals_to_candidates_with_correct_timing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        atoms = [_atom(0, 0, 1000), _atom(1, 1000, 3000), _atom(2, 3000, 6000)]
        calls: list[dict] = []

        async def fake_create(**kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return _fake_completion(
                json.dumps(
                    {
                        "proposals": [
                            {"atom_start": 0, "atom_end": 1, "hint": "a hiring story"},
                            {"atom_start": 2, "atom_end": 2, "hint": "a quick aside"},
                        ]
                    }
                )
            )

        _patch_client(monkeypatch, fake_create)

        candidates = await llm_pass.generate_llm_pass(atoms, Config(), run_dir)

        assert len(calls) == 1
        assert len(candidates) == 2
        assert all(c.generators == ["llm_pass"] for c in candidates)

        first, second = candidates
        assert first.atom_start == 0
        assert first.atom_end == 1
        assert first.start_ms == 0
        assert first.end_ms == 3000  # atoms[1].end_ms
        assert first.generator_hint == "a hiring story"

        assert second.atom_start == 2
        assert second.atom_end == 2
        assert second.start_ms == 3000
        assert second.end_ms == 6000

    @pytest.mark.asyncio
    async def test_uses_configured_generator_model_alias_and_generator_temperature(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from clipper.llm.registry import resolve_model_alias

        run_dir = tmp_path / "run"
        atoms = [_atom(0, 0, 1000)]
        calls: list[dict] = []

        async def fake_create(**kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return _fake_completion(json.dumps({"proposals": []}))

        _patch_client(monkeypatch, fake_create)

        await llm_pass.generate_llm_pass(atoms, Config(llm_generator="gemini-flash"), run_dir)

        assert calls[0]["model"] == resolve_model_alias("gemini-flash")
        assert calls[0]["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_empty_atoms_makes_no_api_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        calls: list[dict] = []

        async def fake_create(**kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return _fake_completion(json.dumps({"proposals": []}))

        _patch_client(monkeypatch, fake_create)

        candidates = await llm_pass.generate_llm_pass([], Config(), run_dir)

        assert candidates == []
        assert calls == []


class TestDefensiveProposalValidation:
    @pytest.mark.asyncio
    async def test_drops_a_proposal_referencing_an_unknown_atom_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        atoms = [_atom(0, 0, 1000), _atom(1, 1000, 2000)]

        async def fake_create(**kwargs: object) -> SimpleNamespace:
            return _fake_completion(
                json.dumps(
                    {
                        "proposals": [
                            {"atom_start": 0, "atom_end": 99, "hint": "hallucinated id"},
                            {"atom_start": 0, "atom_end": 1, "hint": "valid"},
                        ]
                    }
                )
            )

        _patch_client(monkeypatch, fake_create)

        candidates = await llm_pass.generate_llm_pass(atoms, Config(), run_dir)

        assert len(candidates) == 1
        assert candidates[0].generator_hint == "valid"

    @pytest.mark.asyncio
    async def test_drops_a_proposal_with_an_inverted_range(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        atoms = [_atom(0, 0, 1000), _atom(1, 1000, 2000)]

        async def fake_create(**kwargs: object) -> SimpleNamespace:
            return _fake_completion(
                json.dumps({"proposals": [{"atom_start": 1, "atom_end": 0, "hint": "inverted"}]})
            )

        _patch_client(monkeypatch, fake_create)

        candidates = await llm_pass.generate_llm_pass(atoms, Config(), run_dir)

        assert candidates == []


class TestLlmCallFailureDegradesGracefully:
    """blocker 1 fix: `complete_structured` raises `LLMSchemaError` once its
    own retries (M6) are exhausted -- `generate_llm_pass` must not let that
    propagate and crash the whole `generate` stage (spec section 1)."""

    @pytest.mark.asyncio
    async def test_schema_failure_on_the_only_chunk_returns_no_candidates_and_logs_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        atoms = [_atom(0, 0, 1000), _atom(1, 1000, 2000)]

        async def fake_create(**kwargs: object) -> SimpleNamespace:
            # never valid json against `GeneratorOutput` -- exhausts
            # `complete_structured`'s retries and raises `LLMSchemaError`.
            return _fake_completion("not valid json")

        _patch_client(monkeypatch, fake_create)

        candidates = await llm_pass.generate_llm_pass(atoms, Config(), run_dir)

        assert candidates == []

        events = read_events(run_dir, stage="generate")
        assert any(e["event"] == "llm_pass_failed" for e in events)

    @pytest.mark.asyncio
    async def test_schema_failure_on_one_chunk_still_returns_the_other_chunks_candidates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # force chunking (same technique as `TestChunking`).
        monkeypatch.setattr(llm_pass, "_TOKEN_LIMIT_FOR_CHUNKING", 1)

        run_dir = tmp_path / "run"
        atoms = [_atom(0, 0, 1000), _atom(1, 50 * 60_000, 50 * 60_000 + 1000)]

        calls: list[dict] = []

        async def fake_create(**kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            user_prompt = kwargs["messages"][1]["content"]
            if "[0]" in user_prompt:
                # chunk 1 (contains atom 0) always fails schema validation,
                # across all of `complete_structured`'s retry attempts.
                return _fake_completion("not valid json")
            # chunk 2 (contains atom 1) succeeds.
            return _fake_completion(
                json.dumps({"proposals": [{"atom_start": 1, "atom_end": 1, "hint": "second chunk"}]})
            )

        _patch_client(monkeypatch, fake_create)

        candidates = await llm_pass.generate_llm_pass(atoms, Config(), run_dir)

        # chunk 1 contributes nothing (degraded gracefully); chunk 2's
        # proposal still survives -- resilient partial-failure behavior
        # rather than bailing out entirely on one bad chunk.
        assert len(candidates) == 1
        assert candidates[0].generator_hint == "second chunk"

        events = read_events(run_dir, stage="generate")
        assert any(e["event"] == "llm_pass_failed" for e in events)


class TestChunking:
    def test_chunk_atoms_by_time_splits_with_the_configured_overlap(self) -> None:
        atoms = [_atom(0, 0, 1000), _atom(1, 50 * 60_000, 50 * 60_000 + 1000)]

        chunks = llm_pass._chunk_atoms_by_time(
            atoms, chunk_duration_ms=45 * 60_000, overlap_ms=5 * 60_000
        )

        assert len(chunks) == 2
        assert chunks[0] == [atoms[0]]
        assert chunks[1] == [atoms[1]]

    @pytest.mark.asyncio
    async def test_exceeding_the_token_estimate_threshold_triggers_chunking_and_merges_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # force chunking without needing a genuinely huge transcript.
        monkeypatch.setattr(llm_pass, "_TOKEN_LIMIT_FOR_CHUNKING", 1)

        run_dir = tmp_path / "run"
        # two atoms 50 minutes apart -> exactly 2 time-based chunks (see
        # test above), each containing one atom.
        atoms = [_atom(0, 0, 1000), _atom(1, 50 * 60_000, 50 * 60_000 + 1000)]

        calls: list[dict] = []

        async def fake_create(**kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            if len(calls) == 1:
                # chunk 1 proposes {0,0} and (erroneously, as if it saw
                # atom 1 near its chunk boundary) also proposes {0,1} --
                # this duplicate of chunk 2's proposal must be deduped.
                proposals = [
                    {"atom_start": 0, "atom_end": 0, "hint": "first"},
                    {"atom_start": 0, "atom_end": 1, "hint": "overlap-duplicate"},
                ]
            else:
                proposals = [{"atom_start": 0, "atom_end": 1, "hint": "overlap-duplicate"}]
            return _fake_completion(json.dumps({"proposals": proposals}))

        _patch_client(monkeypatch, fake_create)

        candidates = await llm_pass.generate_llm_pass(atoms, Config(), run_dir)

        assert len(calls) == 2  # one real api call per chunk
        # {0,0} and {0,1} survive once each; the second chunk's duplicate
        # {0,1} proposal is dropped.
        ranges = sorted((c.atom_start, c.atom_end) for c in candidates)
        assert ranges == [(0, 0), (0, 1)]
        by_range = {(c.atom_start, c.atom_end): c for c in candidates}
        assert by_range[(0, 1)].generator_hint == "overlap-duplicate"
