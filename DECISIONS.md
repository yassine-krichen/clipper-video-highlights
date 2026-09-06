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

- 2026-09-06: `sha256_llm_cache_key` (spec section 7) joins `model`, `prompt_version`, and
  `input` with a NUL (`\x00`) separator before hashing, rather than the spec's literal
  `sha256(model + prompt_version + input)` naive concatenation. naive concatenation lets
  different inputs collide into the same cache key (e.g. `("ab", "c", "x")` and
  `("a", "bc", "x")` both concatenate to `"abcx"`). NUL cannot appear in any of the three
  components in practice, so the join is unambiguous. `clipper/util/hashing.py`.

- 2026-09-06: `stats.json` and `log.jsonl` (`clipper/util/log.py`) are read/written as plain
  `dict`/`list[dict]`, not pydantic models, unlike every other on-disk artifact (spec section
  3's "everything on disk is one of these serialized"). both files are additive, free-form
  key-value bags written incrementally by many independent call sites across every stage
  (spec section 10); `stats.json`'s nested shape mixes heterogeneous value types (ints,
  floats, strings, per-stage sub-dicts) that don't earn a fixed schema at this stage of the
  build. can be revisited if a stricter shape is needed later.

- 2026-09-06: `new_run_id` (spec section 2.3) uses UTC (`datetime.now(UTC)`) rather than
  local time for the run id timestamp, so run ids are stable and comparable regardless of the
  machine's timezone; the spec's example format doesn't specify which. same applies to
  `log.jsonl`'s per-event `ts` field in `clipper/util/log.py`. this was a proactive choice, not
  one enforced by tooling -- `pyproject.toml`'s `[tool.ruff]` section doesn't configure
  `select`/`lint.select`, so ruff runs its default rule set (E4, E7, E9, F), which does not
  include `DTZ005` (flake8-datetimez is opt-in).

- 2026-09-06: `new_run_id` (spec section 2.3) has a known, accepted collision: two runs of the
  same source file within the same wall-clock second produce an identical run_id (timestamp +
  slug), silently reusing the run directory. low-probability for phase 1's manual, one-run-at-a-
  time CLI usage; not worth solving (e.g. with a sub-second timestamp or a suffix counter) at
  this stage.

- 2026-09-06: `format_ms_as_mmss` (spec section 4.11) does not roll minutes over into an
  `hh:mm:ss` format past 59 minutes -- e.g. `3661000` ms renders as `"61:01"`, not
  `"1:01:01"`. the spec's examples are all under an hour, so this is undetermined by the spec;
  kept simple since `mm:ss` is what section 4.11's `selected.md` output calls for.

- 2026-09-06: `run_probe` (`clipper/stages/probe.py`, spec section 4.1) takes `video_path:
  Path` as a third parameter, deviating from the generic `run_<stage>(run_dir: Path, cfg:
  Config) -> None` signature (spec section 4). probe is the pipeline's entry point -- nothing
  upstream has written the source video's location into the run directory yet -- so the path
  has to be supplied directly. every later stage keeps the generic two-arg signature.

- 2026-09-06: added `Config.max_source_duration_ms` (default 4h in ms) for the probe stage's
  configurable upper duration bound (spec section 4.1: "duration > 4h (configurable)"). named
  distinctly from the existing `min_duration_ms`/`max_duration_ms`, which bound individual
  *clip* length (spec section 2.1) -- reusing those names for the *source video's* duration
  bound would have been ambiguous. the 60s lower bound is spec section 4.1's fixed literal, not
  called out as configurable, so it's a plain constant (`clipper/stages/probe.py`'s
  `_MIN_SOURCE_DURATION_MS`) rather than another config field.

- 2026-09-06: `input.json` (spec section 2.2: "probe result + resolved config") is written as a
  plain dict via `json.dump`, not a pydantic model, unlike most on-disk artifacts (spec section
  3). its shape -- `probe` (ffprobe's raw, arbitrarily-shaped json), `sha256`, `config`
  (`cfg.model_dump()`), `source_path`, `duration_ms` -- mostly wraps other tools' free-form
  output rather than data this codebase defines the shape of, so it doesn't earn a fixed
  pydantic schema, following the same precedent as `stats.json`/`log.jsonl` in
  `clipper/util/log.py`.

- 2026-09-06: `MediaProbeError` (`clipper/media/probe.py`) and `AudioExtractionError`
  (`clipper/media/ffmpeg.py`) live next to the subprocess wrappers that raise them, while
  `InvalidSourceError` (rejection based on the probe *result*: no audio stream, duration out of
  bounds) lives in `clipper/stages/probe.py` since it's a stage-level business rule, not a
  media-layer concern -- probing succeeding is orthogonal to the source being acceptable for
  this pipeline.

- 2026-09-06: reject-path tests for `run_probe` (`tests/integration/test_probe_stage.py`)
  generate real, small synthetic media files with ffmpeg's `lavfi` test sources (`sine=...`,
  `color=...`) at test time rather than mocking ffprobe/ffmpeg or committing tiny fixtures --
  per spec section 13's "never mock ffmpeg or ffprobe" rule, this is real subprocess I/O
  against real (if synthetic) media, exercising the actual rejection logic.

- 2026-09-06: `record_stat` (`clipper/util/log.py`) guards its read-modify-write of
  `stats.json` with an in-process `threading.Lock`, keyed by the file's resolved absolute path
  (a small dict of locks, itself guarded by a meta-lock for safe creation), rather than a
  cross-process file lock (e.g. a `filelock`/`portalocker` dependency). `clipper run` is one
  process per invocation, and the concurrency the spec calls for (section 7: up to 8 concurrent
  judge-stage llm calls, each recording stats) comes from asyncio tasks within that process, not
  separate OS processes -- so an in-process lock fully closes the lost-update race without
  taking on a new dependency or cross-process lock-file lifecycle to manage. the write itself is
  made atomic (write to a temp file in `stats.json`'s directory, then `os.replace()`) so a crash
  mid-write can't leave `stats.json` truncated or corrupted.
