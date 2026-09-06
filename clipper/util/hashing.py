"""content hashing helpers: source file identity (spec section 2.3's
`source.sha256`) and llm cache keys (spec section 7).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024  # 1 MiB; keeps memory flat for multi-hundred-MB video fixtures

# separates the three cache-key components so that e.g. ("ab", "c", "x") and
# ("a", "bc", "x") cannot collide the way naive concatenation would. NUL is
# not expected to appear in any of model / prompt_version / input_payload, so
# this is unambiguous in practice (not an absolute guarantee -- see
# DECISIONS.md). a small, documented improvement over the spec's literal
# `sha256(model + prompt_version + input)` wording.
_CACHE_KEY_SEPARATOR = "\x00"


def sha256_file(path: Path) -> str:
    """sha256 hex digest of a file's bytes, read in fixed-size chunks so
    large video files are never loaded fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_llm_cache_key(model: str, prompt_version: str, input_payload: str) -> str:
    """sha256 hex digest identifying a cached llm call, per spec section 7's
    `sha256(model + prompt_version + input)` cache key formula.

    the NUL-separator join is unambiguous in practice, assuming none of
    `model`, `prompt_version`, or `input_payload` contain a literal NUL
    byte -- `input_payload` in particular is arbitrary external text
    (transcript/prompt content), so this isn't an absolute guarantee.
    """
    joined = _CACHE_KEY_SEPARATOR.join([model, prompt_version, input_payload])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
