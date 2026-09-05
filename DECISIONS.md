# decisions log

one dated line per decision claude code makes that clipper-spec.md did not settle.
reviewed at each phase boundary (spec section 15).

- 2026-09-06: all llm calls (generation, judging, and transcription) route through the
  user's self-hosted freellmapi proxy, a single openai-compatible base_url + api_key,
  instead of separate per-provider keys/sdks as the literal provider aliases in spec
  sections 7 and 9 (`gemini-flash`, `groq-llama`, etc.) assume. `clipper/llm/registry.py`
  (built in a later milestone) resolves model aliases to model-name strings sent to the
  one freellmapi endpoint, not to different providers/base_urls. this also means the
  groq-specific transcriber chunking behavior in section 4.2 will be reimplemented against
  freellmapi's whisper-compatible endpoint rather than the groq sdk.

- 2026-09-06: `Clip.atom_range` (spec section 2.3) is modeled as a `tuple[int, int]` rather
  than `list[int]`, so the "always length 2" constraint is enforced by the type system
  instead of by convention. pydantic serializes tuples to json arrays, so the on-disk shape
  (`"atom_range": [412, 419]`) is unchanged.

- 2026-09-06: content-type scoring weight overrides (spec section 8) are not given exact
  figures, only direction. chose, on top of the default weights
  (hook .22, value .16, emotion .10, curiosity .12, novelty .10, completeness .18, payoff .12):
  - `lecture`: value 0.16 -> 0.20, completeness 0.18 -> 0.22, compensated by lowering
    emotion 0.10 -> 0.06 and curiosity 0.12 -> 0.08 (hook, novelty, payoff unchanged). sums to 1.0.
  - `stream`: emotion 0.10 -> 0.16, completeness 0.18 -> 0.12 (all other dimensions
    unchanged). sums to 1.0.
  `Config.resolved_scoring_weights()` merges the override onto the configured weights and
  re-validates the sum; `Config` also calls it from a `model_validator(mode="after")` at
  construction time, so a bad override fails loud at config-load time, not later
  mid-pipeline when something first calls `resolved_scoring_weights()`.

- 2026-09-06: the example config file (`configs/podcast.yaml`, spec section 5/12) lives in a
  top-level `configs/` directory rather than inside the `clipper/` package, since it's a
  user-facing on-disk artifact passed via `--config`, not importable package code.
