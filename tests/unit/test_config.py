"""tests for clipper/config.py (spec section 2.1, 8)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from clipper.config import Config, ScoringWeights


def test_default_config_matches_spec_defaults() -> None:
    cfg = Config()

    assert cfg.clip_count == 5
    assert cfg.min_duration_ms == 25000
    assert cfg.max_duration_ms == 90000
    assert cfg.content_type == "podcast"
    assert cfg.language == "auto"
    assert cfg.keywords == []
    assert cfg.editorial_instructions == ""
    assert cfg.platform == "shorts"
    assert cfg.allow_multi_segment is False
    assert cfg.transcriber == "groq"
    assert cfg.llm_generator == "gemini-flash"
    assert cfg.llm_judge == "gemini-flash"
    assert cfg.embedder == "local"
    assert cfg.render is False
    assert cfg.captions is False
    assert cfg.reframe == "none"


def test_from_yaml_with_missing_path_uses_defaults() -> None:
    cfg = Config.from_yaml(None)
    assert cfg == Config()


def test_from_yaml_with_nonexistent_file_uses_defaults(tmp_path: Path) -> None:
    cfg = Config.from_yaml(tmp_path / "does_not_exist.yaml")
    assert cfg == Config()


def test_from_yaml_merges_partial_overrides_onto_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "custom.yaml"
    config_path.write_text("clip_count: 8\ncontent_type: lecture\n")

    cfg = Config.from_yaml(config_path)

    assert cfg.clip_count == 8
    assert cfg.content_type == "lecture"
    # unspecified keys keep defaults
    assert cfg.min_duration_ms == 25000
    assert cfg.max_duration_ms == 90000
    assert cfg.platform == "shorts"


def test_scoring_weights_rejects_sum_not_equal_to_one() -> None:
    with pytest.raises(ValidationError, match="must sum to 1.0"):
        ScoringWeights(
            hook=0.5,
            value=0.5,
            emotion=0.5,
            curiosity=0.5,
            novelty=0.5,
            completeness=0.5,
            payoff=0.5,
        )


def test_min_duration_must_be_less_than_max_duration() -> None:
    with pytest.raises(ValidationError, match="min_duration_ms"):
        Config(min_duration_ms=90000, max_duration_ms=25000)


def test_min_duration_equal_to_max_duration_is_rejected() -> None:
    with pytest.raises(ValidationError, match="min_duration_ms"):
        Config(min_duration_ms=50000, max_duration_ms=50000)


def test_max_source_duration_below_60s_floor_is_rejected() -> None:
    with pytest.raises(ValidationError, match="max_source_duration_ms"):
        Config(max_source_duration_ms=59_999)


def test_max_source_duration_at_60s_floor_is_accepted() -> None:
    cfg = Config(max_source_duration_ms=60_000)
    assert cfg.max_source_duration_ms == 60_000


@pytest.mark.parametrize("judge_cap", [0, -1])
def test_judge_cap_below_one_is_rejected(judge_cap: int) -> None:
    with pytest.raises(ValidationError, match="judge_cap"):
        Config(judge_cap=judge_cap)


def test_judge_cap_of_one_is_accepted() -> None:
    cfg = Config(judge_cap=1)
    assert cfg.judge_cap == 1


def test_resolved_scoring_weights_returns_defaults_for_podcast() -> None:
    cfg = Config(content_type="podcast")
    assert cfg.resolved_scoring_weights() == ScoringWeights()


@pytest.mark.parametrize("content_type", ["lecture", "stream"])
def test_resolved_scoring_weights_applies_override_and_sums_to_one(
    content_type: str,
) -> None:
    cfg = Config(content_type=content_type)
    weights = cfg.resolved_scoring_weights()

    assert weights != ScoringWeights()
    total = (
        weights.hook
        + weights.value
        + weights.emotion
        + weights.curiosity
        + weights.novelty
        + weights.completeness
        + weights.payoff
    )
    assert total == pytest.approx(1.0, abs=1e-6)
