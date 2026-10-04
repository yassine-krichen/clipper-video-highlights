"""tests for `clipper/stages/judge.py` (spec section 4.7, section 8).

per spec section 13, only the network call
(`AsyncOpenAI.chat.completions.create`) is mocked -- mirrors
`tests/unit/test_llm_pass_generator.py`'s approach.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipper.config import Config, ScoringWeights
from clipper.llm import cache as llm_cache
from clipper.stages import judge as judge_stage
from clipper.types import JUDGE_DIMENSIONS, Atom, Candidate, JudgeResult, Word
from clipper.util.log import read_events, read_stats

_EQUAL_WEIGHTS = ScoringWeights(
    hook=0.22, value=0.16, emotion=0.10, curiosity=0.12, novelty=0.10, completeness=0.18, payoff=0.12
)


def _atom(atom_id: int, start_ms: int, end_ms: int, text: str | None = None) -> Atom:
    return Atom(
        id=atom_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text=text if text is not None else f"this is atom {atom_id}",
        words=[Word(text=f"atom{atom_id}", start_ms=start_ms, end_ms=end_ms)],
        pause_before_ms=0,
        pause_after_ms=0,
    )


def _candidate(
    candidate_id: str,
    atom_start: int,
    atom_end: int,
    start_ms: int = 0,
    end_ms: int = 0,
    *,
    generator_hint: str | None = None,
) -> Candidate:
    return Candidate(
        id=candidate_id,
        atom_start=atom_start,
        atom_end=atom_end,
        start_ms=start_ms,
        end_ms=end_ms,
        generators=["llm_pass"],
        generator_hint=generator_hint,
        pre_score=0.0,
    )


def _judge_result(
    *,
    dimensions: dict[str, int] | None = None,
    standalone_level: int = 0,
    candidate_id: str = "c1",
) -> JudgeResult:
    return JudgeResult(
        candidate_id=candidate_id,
        dimensions=dimensions or dict.fromkeys(JUDGE_DIMENSIONS, 8),
        standalone_level=standalone_level,
        suggested_atom_start=0,
        suggested_atom_end=0,
        reason="a reason",
        title="a title",
        missing_context=None,
    )


class TestComputeScore:
    def test_standalone_level_0_matches_hand_computed_base_score(self) -> None:
        dims = dict.fromkeys(JUDGE_DIMENSIONS, 8)
        result = _judge_result(dimensions=dims, standalone_level=0)

        score = judge_stage.compute_score(result, _EQUAL_WEIGHTS, keyword_bonus=0.0)

        # base = sum(weight * 8) = 8 * sum(weights) = 8 * 1.0 = 8; score = 80
        assert score == pytest.approx(80.0)

    def test_standalone_level_1_applies_minus_4_adjustment(self) -> None:
        dims = dict.fromkeys(JUDGE_DIMENSIONS, 8)
        result = _judge_result(dimensions=dims, standalone_level=1)

        score = judge_stage.compute_score(result, _EQUAL_WEIGHTS, keyword_bonus=0.0)

        assert score == pytest.approx(80.0 - 4.0)

    def test_standalone_level_2_applies_minus_12_adjustment(self) -> None:
        dims = dict.fromkeys(JUDGE_DIMENSIONS, 8)
        result = _judge_result(dimensions=dims, standalone_level=2)

        score = judge_stage.compute_score(result, _EQUAL_WEIGHTS, keyword_bonus=0.0)

        assert score == pytest.approx(80.0 - 12.0)

    def test_standalone_level_3_short_circuits_to_zero_even_with_high_dimensions(self) -> None:
        # dimensions that would otherwise produce the maximum possible score
        # -- proves the short-circuit actually happens, not just that the
        # formula happens to land near zero.
        dims = dict.fromkeys(JUDGE_DIMENSIONS, 10)
        result = _judge_result(dimensions=dims, standalone_level=3)

        score = judge_stage.compute_score(result, _EQUAL_WEIGHTS, keyword_bonus=5.0)

        assert score == 0.0

    def test_keyword_bonus_is_added_before_clamping(self) -> None:
        dims = dict.fromkeys(JUDGE_DIMENSIONS, 8)
        result = _judge_result(dimensions=dims, standalone_level=0)

        score = judge_stage.compute_score(result, _EQUAL_WEIGHTS, keyword_bonus=3.0)

        assert score == pytest.approx(83.0)

    def test_clamp_catches_score_that_would_exceed_100(self) -> None:
        dims = dict.fromkeys(JUDGE_DIMENSIONS, 10)
        result = _judge_result(dimensions=dims, standalone_level=0)

        # base*10 = 100, plus keyword_bonus (max 5) would be 105 unclamped
        score = judge_stage.compute_score(result, _EQUAL_WEIGHTS, keyword_bonus=5.0)

        assert score == 100.0

    def test_clamp_catches_score_that_would_go_below_zero(self) -> None:
        dims = dict.fromkeys(JUDGE_DIMENSIONS, 1)
        result = _judge_result(dimensions=dims, standalone_level=2)

        # base*10 = 10, minus 12 standalone_adj = -2 unclamped
        score = judge_stage.compute_score(result, _EQUAL_WEIGHTS, keyword_bonus=0.0)

        assert score == 0.0


class TestComputeKeywordBonus:
    def test_no_keywords_configured_returns_zero(self) -> None:
        assert judge_stage.compute_keyword_bonus("some transcript text", []) == 0.0

    def test_matches_documented_formula(self) -> None:
        # 2 of 4 configured keywords appear (case-insensitively) in the text
        text = "we talked about Pricing strategy and hiring mistakes"
        keywords = ["pricing", "hiring", "layoffs", "funding"]

        bonus = judge_stage.compute_keyword_bonus(text, keywords)

        assert bonus == pytest.approx(5.0 * 2 / 4)

    def test_no_keyword_matches_returns_zero(self) -> None:
        bonus = judge_stage.compute_keyword_bonus("nothing relevant here", ["funding", "layoffs"])

        assert bonus == 0.0


class TestCollectPreviousContext:
    def test_includes_atoms_within_the_window_and_excludes_older_ones(self) -> None:
        atoms = [
            _atom(0, 0, 10_000),  # 60_000 - 10_000(end) ... distance from candidate start (70_000) = 70_000-0=70_000 -> excluded
            _atom(1, 20_000, 30_000),  # distance = 70_000 - 20_000 = 50_000 -> included
            _atom(2, 40_000, 60_000),  # distance = 70_000 - 40_000 = 30_000 -> included
            _atom(3, 70_000, 90_000),  # the candidate's own first atom
        ]
        atoms_by_id = {a.id: a for a in atoms}

        context = judge_stage.collect_previous_context(atoms_by_id, atom_start=3, window_ms=60_000)

        assert [a.id for a in context] == [1, 2]

    def test_boundary_atom_exactly_at_the_window_limit_is_included(self) -> None:
        atoms = [
            _atom(0, 10_000, 10_000),  # distance = 70_000 - 10_000 = 60_000 -> included (<=)
            _atom(1, 70_000, 90_000),  # candidate's own first atom
        ]
        atoms_by_id = {a.id: a for a in atoms}

        context = judge_stage.collect_previous_context(atoms_by_id, atom_start=1, window_ms=60_000)

        assert [a.id for a in context] == [0]

    def test_boundary_atom_one_ms_past_the_window_limit_is_excluded(self) -> None:
        atoms = [
            _atom(0, 9_999, 9_999),  # distance = 70_000 - 9_999 = 60_001 -> excluded
            _atom(1, 70_000, 90_000),
        ]
        atoms_by_id = {a.id: a for a in atoms}

        context = judge_stage.collect_previous_context(atoms_by_id, atom_start=1, window_ms=60_000)

        assert context == []

    def test_stops_at_the_start_of_the_transcript(self) -> None:
        atoms = [_atom(0, 0, 1_000), _atom(1, 1_000, 2_000)]
        atoms_by_id = {a.id: a for a in atoms}

        context = judge_stage.collect_previous_context(atoms_by_id, atom_start=1, window_ms=60_000)

        assert [a.id for a in context] == [0]

    def test_no_preceding_atoms_returns_empty_list(self) -> None:
        atoms = [_atom(0, 0, 1_000)]
        atoms_by_id = {a.id: a for a in atoms}

        assert judge_stage.collect_previous_context(atoms_by_id, atom_start=0, window_ms=60_000) == []


class TestCollectFollowingContext:
    def test_includes_atoms_within_the_window_and_excludes_further_ones(self) -> None:
        atoms = [
            _atom(0, 0, 10_000),  # candidate's own last atom
            _atom(1, 10_000, 20_000),  # distance = 20_000 - 10_000 = 10_000 -> included
            _atom(2, 20_000, 40_000),  # distance = 40_000 - 10_000 = 30_000 -> included
            _atom(3, 40_000, 100_000),  # distance = 100_000 - 10_000 = 90_000 -> excluded
        ]
        atoms_by_id = {a.id: a for a in atoms}

        context = judge_stage.collect_following_context(
            atoms_by_id, atom_end=0, window_ms=30_000, max_atom_id=3
        )

        assert [a.id for a in context] == [1, 2]

    def test_boundary_atom_exactly_at_the_window_limit_is_included(self) -> None:
        atoms = [
            _atom(0, 0, 10_000),
            _atom(1, 10_000, 40_000),  # distance = 40_000 - 10_000 = 30_000 -> included (<=)
        ]
        atoms_by_id = {a.id: a for a in atoms}

        context = judge_stage.collect_following_context(
            atoms_by_id, atom_end=0, window_ms=30_000, max_atom_id=1
        )

        assert [a.id for a in context] == [1]

    def test_boundary_atom_one_ms_past_the_window_limit_is_excluded(self) -> None:
        atoms = [
            _atom(0, 0, 10_000),
            _atom(1, 10_000, 40_001),  # distance = 40_001 - 10_000 = 30_001 -> excluded
        ]
        atoms_by_id = {a.id: a for a in atoms}

        context = judge_stage.collect_following_context(
            atoms_by_id, atom_end=0, window_ms=30_000, max_atom_id=1
        )

        assert context == []

    def test_no_following_atoms_returns_empty_list(self) -> None:
        atoms = [_atom(0, 0, 1_000)]
        atoms_by_id = {a.id: a for a in atoms}

        context = judge_stage.collect_following_context(atoms_by_id, atom_end=0, window_ms=30_000, max_atom_id=0)

        assert context == []


@pytest.fixture(autouse=True)
def _freellm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FREELLM_BASE_URL", "https://freellmapi.test/v1")
    monkeypatch.setenv("FREELLM_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def _isolated_global_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_cache, "_GLOBAL_CACHE_DIR", tmp_path / "global_cache")


def _fake_completion(content: str, tokens_in: int = 10, tokens_out: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=tokens_in, completion_tokens=tokens_out),
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


def _valid_llm_response_json(**overrides: object) -> str:
    payload = {
        "dimensions": dict.fromkeys(JUDGE_DIMENSIONS, 8),
        "standalone_level": 0,
        "suggested_atom_start": 0,
        "suggested_atom_end": 0,
        "reason": "clear hook and payoff",
        "title": "a title",
        "missing_context": None,
    }
    payload.update(overrides)
    return json.dumps(payload)


class TestJudgeOneLlmSuccess:
    @pytest.mark.asyncio
    async def test_builds_judge_result_and_computes_score(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        atoms = [_atom(0, 0, 1_000), _atom(1, 1_000, 2_000)]
        atoms_by_id = {a.id: a for a in atoms}
        candidate = _candidate("c1", 0, 1, start_ms=0, end_ms=2_000)

        async def fake_create(**kwargs: object) -> SimpleNamespace:
            return _fake_completion(_valid_llm_response_json())

        _patch_client(monkeypatch, fake_create)

        judged, used_fallback = await judge_stage._judge_one(
            candidate, atoms_by_id, max_atom_id=1, cfg=Config(), run_dir=run_dir,
            weights=_EQUAL_WEIGHTS, semaphore=asyncio.Semaphore(8),
        )

        assert used_fallback is False
        assert judged.judge.candidate_id == "c1"
        assert judged.judge.dimensions == dict.fromkeys(JUDGE_DIMENSIONS, 8)
        assert judged.score == pytest.approx(80.0)


class TestJudgeOneLlmSchemaFailureFallback:
    @pytest.mark.asyncio
    async def test_fallback_judge_result_is_constructed_exactly_as_documented(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        atoms = [_atom(0, 0, 1_000), _atom(1, 1_000, 2_000)]
        atoms_by_id = {a.id: a for a in atoms}
        candidate = _candidate("c1", 0, 1, start_ms=0, end_ms=2_000, generator_hint="a hiring story")

        async def fake_create(**kwargs: object) -> SimpleNamespace:
            # never valid json against `JudgeLLMResponse` -- exhausts
            # `complete_structured`'s retries and raises `LLMSchemaError`.
            return _fake_completion("not valid json")

        _patch_client(monkeypatch, fake_create)

        judged, used_fallback = await judge_stage._judge_one(
            candidate, atoms_by_id, max_atom_id=1, cfg=Config(), run_dir=run_dir,
            weights=_EQUAL_WEIGHTS, semaphore=asyncio.Semaphore(8),
        )

        assert used_fallback is True
        assert judged.judge.dimensions == dict.fromkeys(JUDGE_DIMENSIONS, 5)
        assert judged.judge.standalone_level == 2
        assert judged.judge.reason == "judge failed"
        assert judged.judge.suggested_atom_start == candidate.atom_start
        assert judged.judge.suggested_atom_end == candidate.atom_end
        assert judged.judge.title == "a hiring story"  # falls back to generator_hint

        # standalone_level 2 (not 3) -- normal score computation proceeds,
        # not the standalone_level == 3 short-circuit to 0.0.
        expected_score = judge_stage.compute_score(judged.judge, _EQUAL_WEIGHTS, keyword_bonus=0.0)
        assert judged.score == pytest.approx(expected_score)
        assert judged.score == pytest.approx(5.0 * 10 - 12.0)

        events = read_events(run_dir, stage="judge")
        assert any(e["event"] == "judge_fallback_used" for e in events)

    @pytest.mark.asyncio
    async def test_fallback_title_placeholder_when_no_generator_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        atoms = [_atom(0, 0, 1_000)]
        atoms_by_id = {a.id: a for a in atoms}
        candidate = _candidate("c1", 0, 0, generator_hint=None)

        async def fake_create(**kwargs: object) -> SimpleNamespace:
            return _fake_completion("not valid json")

        _patch_client(monkeypatch, fake_create)

        judged, _ = await judge_stage._judge_one(
            candidate, atoms_by_id, max_atom_id=0, cfg=Config(), run_dir=run_dir,
            weights=_EQUAL_WEIGHTS, semaphore=asyncio.Semaphore(8),
        )

        assert judged.judge.title == "(judge failed)"


class TestJudgeConcurrencyLimit:
    def test_never_exceeds_8_concurrent_judge_calls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # NOT `async def` / `@pytest.mark.asyncio`: `run_judge` is a plain
        # sync function that drives its own event loop internally via
        # `asyncio.run` (spec section 4.7's stage signature is sync). an
        # async test here would already be running inside pytest-asyncio's
        # own loop, and `asyncio.run` refuses to nest.
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        atoms = [_atom(i, i * 1_000, i * 1_000 + 1_000) for i in range(20)]
        candidates = [_candidate(f"c{i}", i, i, start_ms=i * 1_000, end_ms=i * 1_000 + 1_000) for i in range(20)]

        (run_dir / "atoms.json").write_text(json.dumps([a.model_dump() for a in atoms]))
        (run_dir / "candidates.json").write_text(json.dumps([c.model_dump() for c in candidates]))

        in_flight = 0
        max_in_flight = 0
        lock = asyncio.Lock()

        async def fake_create(**kwargs: object) -> SimpleNamespace:
            nonlocal in_flight, max_in_flight
            async with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            async with lock:
                in_flight -= 1
            return _fake_completion(_valid_llm_response_json())

        _patch_client(monkeypatch, fake_create)

        judge_stage.run_judge(run_dir, Config())

        assert max_in_flight <= 8


class TestRunJudgeStageWiring:
    def test_reads_candidates_and_atoms_writes_judged_and_populates_log_and_stats(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        atoms = [_atom(0, 0, 1_000), _atom(1, 1_000, 2_000)]
        candidates = [_candidate("c1", 0, 1, start_ms=0, end_ms=2_000)]

        (run_dir / "atoms.json").write_text(json.dumps([a.model_dump() for a in atoms]))
        (run_dir / "candidates.json").write_text(json.dumps([c.model_dump() for c in candidates]))

        async def fake_create(**kwargs: object) -> SimpleNamespace:
            return _fake_completion(_valid_llm_response_json())

        _patch_client(monkeypatch, fake_create)

        judge_stage.run_judge(run_dir, Config())

        judged_raw = json.loads((run_dir / "judged.json").read_text())
        assert len(judged_raw) == 1
        assert judged_raw[0]["judge"]["candidate_id"] == "c1"
        assert judged_raw[0]["score"] == pytest.approx(80.0)

        events = read_events(run_dir, stage="judge")
        event_names = [e["event"] for e in events]
        assert "started" in event_names
        assert "judged" in event_names

        stats = read_stats(run_dir)
        assert stats["candidates"]["judged"] == 1
        assert stats["judge"]["fallback_count"] == 0
        assert "wall_ms" in stats["stages"]["judge"]

    def test_empty_candidates_writes_empty_judged_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        atoms = [_atom(0, 0, 1_000)]

        (run_dir / "atoms.json").write_text(json.dumps([a.model_dump() for a in atoms]))
        (run_dir / "candidates.json").write_text(json.dumps([]))

        async def fake_create(**kwargs: object) -> SimpleNamespace:
            raise AssertionError("no candidates -- the llm should never be called")

        _patch_client(monkeypatch, fake_create)

        judge_stage.run_judge(run_dir, Config())

        judged_raw = json.loads((run_dir / "judged.json").read_text())
        assert judged_raw == []
