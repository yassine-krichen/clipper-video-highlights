"""run configuration: pydantic-settings model loadable from yaml + env vars
(spec section 2.1) and scoring weights (spec section 8).

config validation fails loudly and immediately at construction time, per the
"validate config at startup" rule -- never a runtime surprise mid-run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ContentType = Literal["podcast", "interview", "lecture", "stream", "auto"]
Platform = Literal["shorts", "tiktok", "reels", "generic"]
Transcriber = Literal["groq", "faster_whisper"]
Embedder = Literal["local", "api"]
Reframe = Literal["none", "center", "face"]

_WEIGHT_SUM_TOLERANCE = 1e-6


class ScoringWeights(BaseModel):
    """weights for the base score formula (spec section 8). must sum to 1.0."""

    hook: float = 0.22
    value: float = 0.16
    emotion: float = 0.10
    curiosity: float = 0.12
    novelty: float = 0.10
    completeness: float = 0.18
    payoff: float = 0.12

    @model_validator(mode="after")
    def _validate_sum_to_one(self) -> ScoringWeights:
        total = (
            self.hook
            + self.value
            + self.emotion
            + self.curiosity
            + self.novelty
            + self.completeness
            + self.payoff
        )
        if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"scoring weights must sum to 1.0, got {total}")
        return self


DEFAULT_SCORING_WEIGHTS = ScoringWeights()

# per-content-type partial overrides, merged onto DEFAULT_SCORING_WEIGHTS
# (spec section 8: "lecture raises value and completeness, stream raises
# emotion and lowers completeness"). exact figures are not given by the spec;
# chosen to satisfy the direction while keeping the merged weights summing to
# 1.0 -- see DECISIONS.md for the reasoning.
CONTENT_TYPE_WEIGHT_OVERRIDES: dict[str, dict[str, float]] = {
    "lecture": {
        "value": 0.20,
        "completeness": 0.22,
        "emotion": 0.06,
        "curiosity": 0.08,
    },
    "stream": {
        "emotion": 0.16,
        "completeness": 0.12,
    },
}


class Config(BaseSettings):
    """resolved run configuration (spec section 2.1). loadable from a yaml
    file via `Config.from_yaml`, and from env vars via pydantic-settings'
    default behavior (prefix `CLIPPER_`)."""

    model_config = SettingsConfigDict(env_prefix="CLIPPER_")

    clip_count: int = 5
    min_duration_ms: int = 25000
    max_duration_ms: int = 90000
    content_type: ContentType = "podcast"
    language: str = "auto"
    keywords: list[str] = []
    editorial_instructions: str = ""
    platform: Platform = "shorts"
    allow_multi_segment: bool = False
    transcriber: Transcriber = "groq"
    llm_generator: str = "gemini-flash"
    llm_judge: str = "gemini-flash"
    embedder: Embedder = "local"
    render: bool = False
    captions: bool = False
    reframe: Reframe = "none"
    scoring_weights: ScoringWeights = DEFAULT_SCORING_WEIGHTS

    @field_validator("max_duration_ms")
    @classmethod
    def _validate_duration_bounds(cls, max_duration_ms: int, info: ValidationInfo) -> int:
        min_duration_ms = info.data.get("min_duration_ms")
        if min_duration_ms is not None and min_duration_ms >= max_duration_ms:
            raise ValueError(
                f"min_duration_ms ({min_duration_ms}) must be less than "
                f"max_duration_ms ({max_duration_ms})"
            )
        return max_duration_ms

    @model_validator(mode="after")
    def _validate_resolved_scoring_weights(self) -> Config:
        # forces the content_type override merge (if any) to happen now, so a
        # bad weights override is caught at config-load time, not later
        # mid-pipeline when resolved_scoring_weights() first gets called.
        self.resolved_scoring_weights()
        return self

    @classmethod
    def from_yaml(cls, path: Path | None) -> Config:
        """load config from a yaml file, merging onto defaults. a missing
        file, a missing path, or missing keys within the file all just fall
        back to defaults -- only present keys override."""
        if path is None or not path.exists():
            return cls()

        raw = yaml.safe_load(path.read_text()) or {}
        return cls(**raw)

    def resolved_scoring_weights(self) -> ScoringWeights:
        """the configured scoring weights with the content_type override
        applied on top, if one exists for this content_type."""
        override = CONTENT_TYPE_WEIGHT_OVERRIDES.get(self.content_type)
        if override is None:
            return self.scoring_weights

        merged = self.scoring_weights.model_dump()
        merged.update(override)
        return ScoringWeights(**merged)
