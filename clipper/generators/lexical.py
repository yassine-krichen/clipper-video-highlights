"""g3 `lexical` generator (spec section 4.5): deterministic, no llm.

for every atom whose text contains a hook pattern
(`clipper/generators/hooks.yaml`), emits two candidates: the window from
that atom forward to the first atom whose end satisfies `min_duration_ms`,
and one extended as far as it fits within `max_duration_ms`.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from clipper.config import Config
from clipper.generators.base import (
    build_candidate,
    max_duration_window_end,
    min_duration_window_end,
)
from clipper.types import Atom, Candidate

_HOOKS_YAML_PATH = Path(__file__).parent / "hooks.yaml"
_DEFAULT_LANGUAGE = "en"


def _load_hooks(language: str) -> tuple[list[str], re.Pattern[str]]:
    """load the hook phrase list and currency regex for `language` from
    `hooks.yaml`, falling back to `_DEFAULT_LANGUAGE` ("en") when `language`
    has no entry -- covers both `cfg.language == "auto"` and any language
    without a shipped pattern set (phase 1 only ships English, see
    `hooks.yaml`'s module comment)."""
    data = yaml.safe_load(_HOOKS_YAML_PATH.read_text(encoding="utf-8"))
    lang_data = data.get(language, data[_DEFAULT_LANGUAGE])
    phrases = [phrase.lower() for phrase in lang_data["phrases"]]
    currency_pattern = re.compile(lang_data["currency_pattern"], re.IGNORECASE)
    return phrases, currency_pattern


def _atom_matches_hook(atom_text: str, phrases: list[str], currency_pattern: re.Pattern[str]) -> bool:
    """case-insensitive substring match against `phrases`, or a currency
    regex match -- deliberately simple (no stemming/tokenization), matching
    the spec's "numbers with currency, etc." framing as a lightweight
    heuristic rather than full nlp."""
    lowered = atom_text.lower()
    if any(phrase in lowered for phrase in phrases):
        return True
    return currency_pattern.search(atom_text) is not None


def generate_lexical(atoms: list[Atom], cfg: Config) -> list[Candidate]:
    """g3: see module docstring."""
    phrases, currency_pattern = _load_hooks(cfg.language)

    candidates: list[Candidate] = []
    for index, atom in enumerate(atoms):
        if not _atom_matches_hook(atom.text, phrases, currency_pattern):
            continue

        min_end_atom = min_duration_window_end(atoms, index, cfg.min_duration_ms)
        if min_end_atom is not None:
            candidates.append(build_candidate(atom, min_end_atom, "lexical"))

        max_end_atom = max_duration_window_end(atoms, index, cfg.max_duration_ms)
        if max_end_atom is not None:
            candidates.append(build_candidate(atom, max_end_atom, "lexical"))

    return candidates
