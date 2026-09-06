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

- 2026-09-06: milestone m4 (transcribers, spec section 4.2) judgment calls:
  - `chunk_wav` (`clipper/transcribers/groq.py`) uses ffmpeg subprocess calls to cut
    fixed-size, overlapping wav slices to temp files -- consistent with the rest of the
    codebase's approach to audio (`clipper/media/ffmpeg.py`), and avoids a second audio
    i/o dependency (`wave` module would still need separate resampling/format handling
    groq's endpoint expects). chunk `i`'s start offset is `i * (chunk_duration_ms -
    overlap_ms)`, so consecutive chunks overlap by exactly `overlap_ms` regardless of how
    many chunks the source needs; audio no longer than `chunk_duration_ms` skips chunking
    entirely and is transcribed in a single call, per spec ("groq caps request size").
  - `stitch_chunks` matches dropped overlap words purely on `(start_ms, end_ms)` within a
    +/-100ms tolerance (spec is explicit: match on timing, not text) -- a word inside the
    `[chunk_offset_ms, chunk_offset_ms + overlap_ms)` absolute-time zone is dropped only if
    a word already emitted from the previous chunk's tail matches its timing within
    tolerance; words in the zone with no timing match (e.g. the previous chunk's
    transcription genuinely stopped a bit short of the true overlap boundary) are kept, not
    dropped by zone membership alone.
  - `env_prefix`-based `pydantic-settings` config (`Config`) is not extended to own
    `FREELLM_BASE_URL`/`FREELLM_API_KEY`: those are read directly via `os.environ.get` in
    `clipper/transcribers/groq.py`'s `_build_client`, raising a clear `RuntimeError` if
    either is unset. they're infrastructure/secrets for the freellmapi proxy, not run
    configuration a yaml file or `Config.from_yaml` should ever set -- keeping them out of
    `Config` also means `cfg.model_dump()` (written into every run's `input.json`, spec
    section 4.1) never risks including them. `clipper/llm/client.py` (a later milestone)
    should read them the same way for consistency.
  - GroqTranscriber -> FasterWhisperTranscriber fallback (`clipper/stages/transcribe.py`)
    treats an error as "non-retryable" (spec section 4.2's exact fallback trigger) via a
    substring heuristic on the exception message: missing/invalid freellmapi config or
    credentials (`FREELLM_BASE_URL`/`FREELLM_API_KEY`, "authentication", "unauthorized",
    "invalid_api_key", 401/403) or a bad model name (404/"model_not_found"). rate limits,
    timeouts, and 5xx responses are deliberately excluded -- those are transient, and
    silently swapping to a slower, lower-quality local model on a transient failure would
    be a worse outcome than surfacing the error. this is a heuristic over message text, not
    a typed exception hierarchy from the `openai` sdk, since freellmapi's proxying may not
    preserve the sdk's normal exception subclasses faithfully.
  - `faster-whisper` gpu/cpu model selection (`clipper/transcribers/faster_whisper.py`)
    uses `ctranslate2.get_cuda_device_count()`, not `torch.cuda.is_available()` -- torch is
    only a transitive dependency here (via `sentence-transformers`) and its installed cuda
    build has no bearing on what ctranslate2 (faster-whisper's actual inference backend)
    can use. this only reflects driver-level gpu presence, though: on this dev machine,
    `get_cuda_device_count()` reports a gpu, but the cuda runtime it needs
    (`cublas64_12.dll`) isn't loadable, and ctranslate2 only surfaces that as a
    `RuntimeError` on first real inference, not at model-load time. `FasterWhisperTranscriber.transcribe`
    catches that specific failure (message contains `cublas`/`cudnn`/`.dll`/`.so`) and
    retries once on the cpu model (`small`) -- confirmed against a real run against
    `tests/fixtures/sample_3min.mp4`'s audio on this machine.
  - also observed on this dev machine: the `vad_filter=True` spec default (silero vad via
    onnxruntime, bundled in `faster-whisper`) returns zero speech segments for real,
    non-silent, plainly-transcribable audio -- confirmed independently of this codebase by
    calling the vad model directly and inspecting raw speech probabilities (max ~0.33,
    never crossing its 0.5 threshold), and by re-running the same audio through
    `WhisperModel.transcribe` with `vad_filter=False`, which works normally. this looks like
    an onnxruntime/silero-vad-onnx-asset environment incompatibility upstream of this
    codebase (`faster-whisper==1.2.1`'s bundled `silero_vad_v6.onnx`), not a bug in
    `FasterWhisperTranscriber`. rather than silently ship an empty transcript,
    `transcribe()` retries once with `vad_filter=False` whenever the first
    (spec-mandated) attempt returns zero words for non-empty audio -- whisper's decoder
    still skips genuine silence on its own, just without vad's pre-filtering benefit. this
    fallback was exercised and confirmed working end-to-end against the real fixture on
    this machine (`tests/integration/test_faster_whisper_transcriber.py`, marked `slow`).
  - added a `slow` pytest marker (`pyproject.toml`) for
    `tests/integration/test_faster_whisper_transcriber.py`, since it loads a real local
    whisper model (first run downloads it) and runs real cpu inference -- too heavy for a
    fast default test loop. `uv run pytest -m "not slow"` deselects it; the milestone's
    full suite (including this test) was still run and confirmed passing at least once.
  - `chunk_wav`'s multi-chunk-splitting test (`tests/unit/test_chunk_wav.py`) uses a much
    shorter `chunk_duration_ms=30_000` against the real `sample_3min.mp4` fixture (~144s)
    rather than generating a synthetic long wav via ffmpeg `lavfi` -- the real fixture
    already exists, is committed, and is short enough that a 30s chunk size forces several
    real chunks without needing new fixture generation machinery, while still running real,
    unmocked ffmpeg/ffprobe calls per spec section 13.

- 2026-09-06: M4 review follow-up: `FREELLM_BASE_URL`/`FREELLM_API_KEY` (read via
  `os.environ.get` in `clipper/transcribers/groq.py`'s `_build_client`) won't actually be
  populated from a `.env` file until something calls `load_dotenv()` (or equivalent) in the
  CLI entrypoint -- out of scope for M4, and deliberately not added here since `.env`
  loading/CLI wiring belongs to the later CLI-wiring milestone (M15). flagging now so that
  milestone doesn't forget it.

- 2026-09-06: milestone m5 (`clipper/stages/atomize.py`, spec section 4.3) judgment calls:
  - pipeline order and iteration: the spec lists the four rules as
    `split -> merge-short -> split-long -> compute-pauses` without saying
    whether any step needs more than one pass. `split_into_atoms` runs once
    (each word is visited exactly once, unambiguous). `merge_short_atoms`
    runs to a fixed point (`while changed:` around a full left-to-right
    pass): merging two atoms can make the merged atom's new neighbour need
    re-checking (e.g. three consecutive 1-word atoms need two merge passes
    to end up as one atom), so a single pass under-merges in that case.
    **(corrected 2026-09-06, post-review, round 1)** `split_long_atoms` was
    originally believed to be safe as a single pass, on the theory that
    splitting at the single largest internal gap guarantees each resulting
    half can't contain a *larger* gap than the one just used -- but that
    only bounds the gap used for the *next* split, it says nothing about
    whether either half is actually now under `_MAX_ATOM_MS`. a half can
    still be well over 25s and need to be split again on its own internal
    gaps (reproduced with a 4-word atom whose single largest gap splits it
    into a 100ms atom and a 31.1s atom -- still way over the limit).
    `split_long_atoms` now runs each atom through
    `_split_atom_to_fixed_point`, a recursive split that re-checks every
    resulting piece against `_MAX_ATOM_MS` and keeps splitting until every
    piece is under the limit or hits the terminal no-valid-split-point case
    (see below).

    **(corrected 2026-09-06, post-review, round 2 -- later found to be
    itself flawed and superseded, see round 3 below)** round 1's fix
    introduced a separate problem: because `merge_short_atoms` runs
    *before* `split_long_atoms` in the listed order, and `split_long_atoms`
    could itself create a new sub-3-word fragment with no pause protection
    (by splitting at whichever gap happened to be largest, with no regard
    for what it would leave behind), a single top-to-bottom pass of the
    whole pipeline could leave that fragment un-merged forever. the round 2
    fix made `atomize_words` alternate `merge_short_atoms` and
    `split_long_atoms` in a `while` loop at the pipeline level, re-running
    each after the other changed anything, until one full round of both
    produced no change, plus cycle detection (tracking every post-merge
    state already produced; if a merge ever reproduced one, the loop
    stopped and kept the merged, possibly-still-over-25s, form) to handle
    adversarial atoms where split and merge would otherwise oscillate
    forever.

    **(corrected 2026-09-06, post-review, round 3 -- replaces round 2
    entirely)** round 2's alternating-loop/cycle-detection fix was itself a
    worse bug: the last-atom-short backward-merge (see below), when repeated
    across fixed-point rounds, had no bound on how much it could consume.
    concretely: a split-created unprotected fragment would get merged
    backward by `merge_short_atoms`; if the *whole* atom (including
    already-compliant content merged in from far earlier) got reproduced by
    that merge, the round-2 cycle detector would settle on that merged
    state -- meaning an atom could absorb arbitrarily much unrelated,
    already-compliant preceding content into one giant blob, not just
    "slightly over 25s" as originally believed. confirmed repro: a 100-word,
    ~20s benign block immediately followed by an 8-word tail whose single
    largest internal gap sat right before its last word (creating a 1-word
    unprotected fragment if cut there) collapsed into one 25,690ms atom that
    absorbed the entire benign block -- nearly 3x the number of words that
    should ever have been touched by the tail's own split/merge resolution.

    the actual root cause was narrower than round 2's fix addressed:
    `split_long_atoms`'s split-point selection picked the single largest
    gap with no awareness of the word-count/pause-protection rule, so it
    could freely create an unprotected fragment and hand the problem off to
    a *global* merge pass that had no way to know the fragment was scoped
    to just that one split decision. the round 3 fix makes split-point
    selection itself aware of the constraint (`_select_split_index`,
    `_creates_unprotected_short_child`): candidate gaps are tried
    largest-first, and a candidate is skipped -- not accepted and
    patched up later -- if it would leave either resulting child under
    `_MIN_ATOM_WORDS` words *and* the split gap itself is under
    `_PAUSE_SPLIT_MS` (the same exception `merge_short_atoms` already
    grants). `split_long_atoms` therefore never creates a new unprotected
    short fragment in the first place, so there is nothing left for a merge
    pass to clean up, no alternation between `merge_short_atoms` and
    `split_long_atoms` is needed, and the round-2 cycle-detection machinery
    (`seen_merged_states`, `_atoms_signature`, the `atomize_words`-level
    `while` loop) has been removed entirely. `atomize_words` is back to a
    single straight-line pass: `split_into_atoms -> merge_short_atoms ->
    split_long_atoms -> compute_pauses -> _assign_ids`.

    if *every* candidate gap in an atom would create an unprotected short
    fragment (or there's no internal gap at all), `_select_split_index`
    returns `None` and the atom is left as a single, local, well-scoped
    exception to the 25s limit -- over the limit, yes, but strictly because
    that atom's own content structure makes it truly unsplittable without
    violating the word-count rule, never by absorbing any neighbouring
    atom's content. this is a materially different (and much smaller)
    exception than round 2's: it can only ever affect the one atom whose own
    gaps are all disqualified, with a size bounded by whatever content
    `split_into_atoms` originally grouped into that one atom -- it cannot
    grow across merge/split rounds, and no unrelated neighbour is ever
    touched. documented and tested explicitly (round 1's repro -- 5 words,
    total span 25,100ms, every one of its 4 internal gaps under 700ms and
    disqualified on word-count grounds -- now resolves directly via
    `split_long_atoms` alone, no merge/cycle involved:
    `test_atom_where_every_split_point_would_create_an_unprotected_fragment_is_left_unsplit`;
    the round 3 100-word-benign-block-plus-tail repro is
    `test_benign_atom_preceding_an_unsplittable_tail_is_not_absorbed_into_a_giant_merged_atom`,
    both in `tests/unit/test_atomize.py`).

    the last-atom-short backward-merge exception (see below) was
    re-examined given this redesign and kept as-is: it's sound on its own
    merits, independent of this bug -- the round 3 runaway was caused by
    `split_long_atoms` handing merge an unprotected fragment it had no way
    to scope correctly, not by the backward-merge rule itself. with
    split-point selection now guaranteed to never create such a fragment,
    the backward merge only ever fires on fragments `split_into_atoms`
    produced directly from the original transcript, which is exactly the
    case it was designed for.

    order-wise, merge-short still runs *before* split-long (as listed):
    merging first means a long atom created by merging fragments together
    still gets its long-atom check in the same pass; splitting first then
    merging would risk re-fusing a freshly split long atom back together if
    either half were short, which would defeat the point of the split.
    `compute_pauses` and final id assignment always run last, once, since
    they depend on final neighbour relationships.
  - sentence-final punctuation matching (`_ends_sentence`): a word ends a
    sentence if, after stripping trailing closing quote/paren/bracket
    characters (`"'"`)]}`, it ends in `.`, `?`, or `!`. this covers common
    cases like `"Really?"` and `(fine.)` without building real sentence
    boundary detection (e.g. no special-casing of abbreviations like "Mr."
    or decimal numbers like "3.5" -- those will incorrectly end an atom).
    good enough for a deterministic, no-llm first pass; a mis-split here
    only produces a slightly-too-short atom, which `merge_short_atoms` may
    then repair anyway if the fragment is under 3 words.
  - last atom in the transcript is short (`merge_short_atoms`): the spec's
    "merge shorter atoms into the following atom" rule has no following
    atom for the very last one. chose to merge it backward into the
    previous (already-finalized) atom instead of leaving it as an
    under-length trailing atom. **(corrected 2026-09-06, post-review)**
    this backward merge now *does* consult the same `>=700ms` pause
    exception every forward merge respects: if the previous atom is
    separated from the short last atom by a real pause, forcing them
    together would be exactly the kind of unnatural fusion the exception
    exists to prevent, regardless of which direction the merge runs. if the
    pause is `>=700ms`, the short last atom is left as its own trailing
    atom with fewer than 3 words -- that's fine, since the whole point of
    the pause exception (spec bullet 3) is that a short-but-pause-protected
    atom is allowed to stay short. documented and tested explicitly
    (`test_last_atom_short_with_nothing_following_merges_backward` for the
    merge case, `test_last_atom_short_separated_by_large_pause_stays_unmerged`
    for the new pause-protected case).
  - long atom with no internal pause (`split_long_atoms`): if every word in
    an atom is back-to-back (e.g. a single very long word, or a run of
    zero-gap words) there's no internal gap to split at. left as-is rather
    than forcing an arbitrary mid-word or mid-atom cut with no principled
    location -- documented and tested
    (`test_long_atom_with_no_internal_pause_is_left_as_is`).
  - `atoms.json` shape: a plain json array of `Atom` objects (via
    `json.dumps([atom.model_dump() for atom in atoms], indent=2)`), matching
    the spec's `Atom[]` (section 2.2) literally, rather than wrapping in an
    envelope object -- no other stage's output needs an envelope, and there
    is no cross-cutting metadata (page count, provider info, etc.) that
    would justify one here, unlike `transcript.json`'s `Transcript` object
    which already carries `language`/`provider`/`model` alongside its words.

- 2026-09-06: milestone m-llm-layer (`clipper/llm/registry.py`, `cache.py`,
  `client.py`, spec section 7) judgment calls:
  - `registry.py`'s `_MODEL_ALIASES` dict maps each alias
    (`gemini-flash`/`gemini-pro`/`groq-llama`/`claude-sonnet`) to a
    PLACEHOLDER model-name string following each provider's own public
    naming convention (e.g. `"gemini-2.0-flash"`, `"llama-3.3-70b-versatile"`).
    these are not verified against freellmapi's actual model catalog -- there
    is no way to do that without calling the real, live proxy, which the
    test suite must never do (spec section 13). must be checked/adjusted
    against freellmapi's real catalog before any of the four aliases are
    used for a real (non-mocked) call.
  - cache lookup order (spec section 7 doesn't specify): `cache.py`'s
    `get_cached_response` checks the run-local `run_dir/llm_cache/<key>.json`
    first, then the global `~/.clipper/cache/<key>.json`. a global-only hit
    is copied into run-local before returning, so every run directory stays
    self-describing for later inspection/debugging without needing to
    cross-reference the global cache.
  - `client.py`'s cache key input component is `system_prompt + user_prompt`
    -- everything that affects the model's output -- deliberately excluding
    `temperature`, per the spec's literal `sha256(model + prompt_version +
    input)` formula (section 7), which has no temperature term.
  - `_estimate_cost_usd` (`client.py`) is a flat placeholder: `(tokens_in +
    tokens_out) / 1000 * 0.0005`, input and output priced identically. real
    per-model pricing needs calibration against freellmapi's actual
    billing/passthrough behavior once that's knowable -- not guessable from
    this codebase alone.
  - `client.py`'s `_build_client` (an `AsyncOpenAI` client pointed at
    `FREELLM_BASE_URL`/`FREELLM_API_KEY`) is a small, deliberate duplicate of
    `clipper/transcribers/groq.py`'s sync `_build_client`, not factored into
    a shared helper: both are ~6 lines, one sync (groq's transcription sdk
    surface) and one async (spec section 4.7's up to 8 concurrent judge
    calls), and a new shared module for this isn't listed in spec section
    6's repository layout. the duplication is small enough that the extra
    indirection of a shared module wasn't judged worth it; revisit if a
    third call site needs the same construction logic.
  - `complete_structured`'s retry loop treats "up to 3 times" (spec section
    7) as 3 total attempts (not 3 retries after an initial attempt, i.e. 4
    total) -- `_MAX_ATTEMPTS = 3`. `LLMSchemaError` is raised only after all
    3 attempts fail json/schema validation; it is NOT the judge-specific
    "dimensions all 5, standalone_level 2, reason 'judge failed'" fallback
    from spec section 7 -- that fallback is judge-stage business logic and
    belongs in the (later) judge stage, which will catch `LLMSchemaError`
    and apply it. the generic client only raises.
  - each llm call is recorded under `stats.json`'s `llm_calls` section keyed
    by `f"{prompt_version}_{uuid4().hex[:12]}"` rather than by
    `prompt_version` alone -- `record_stat`'s `(section, key)` pair
    overwrites on collision (by design, see the m2 entry below), so a stable
    key per prompt version would silently drop every call after the first
    within the same run. a cache hit still records a `$0`/0-token entry
    (spec section 7's cost accounting should reflect every call that
    "happened", cache hit or not).

- 2026-09-06: M6 review follow-up (`clipper/llm/client.py`, `cache.py`, spec section 7):
  - blocking fix -- `_record_call_stat` was only called once, in
    `complete_structured`'s success branch, using only the LAST attempt's
    token counts. every earlier FAILED attempt in a retry sequence made a
    real, billed api call whose tokens/cost were never recorded, and total
    exhaustion (`LLMSchemaError`) recorded zero stats for any of the (up to
    `_MAX_ATTEMPTS`) real calls made. fixed by moving the stat recording
    inside the retry loop itself: each attempt's `usage` is read and
    `_record_call_stat` called immediately after that attempt's api call
    returns, before the json/schema validation that decides whether to
    retry -- so a stat is recorded for every real call, success or failure,
    with no separate "record on final exhaustion" step needed (each attempt
    already recorded itself as it happened). confirmed via
    `TestCompleteStructuredRetryThenSucceed` (asserts 2 recorded stats: the
    failed attempt + the success) and `TestCompleteStructuredRetryExhausted`
    (asserts 3 recorded stats, all failed) in `tests/unit/test_llm_client.py`.
  - high-severity fix -- `_json_schema_response_format` passed
    `response_model.model_json_schema()` verbatim with `"strict": True`, but
    pydantic's `model_json_schema()` never emits `additionalProperties:
    false` (top-level or under `$defs`) and excludes optional/defaulted
    fields from `required` -- both are hard requirements of
    OpenAI-compatible strict structured-output validation, and the schema
    would very likely have been rejected by the real endpoint the first time
    a response model had a nested submodel or an optional field (exactly
    what M7's `JudgeResult`/generator response models need). fixed with a
    new `_make_strict_schema`/`_strict_schema_node`/`_allow_null_type` schema
    post-processing pass (`clipper/llm/client.py`) that recursively walks
    every object-type node -- including everything under `$defs` -- setting
    `additionalProperties: false` and promoting every property into
    `required`, adding `null` as an allowed type/branch (via a type-array,
    or an `anyOf` wrapper for bare `$ref` nodes) for any property pydantic
    did NOT originally mark required. verified structurally (no live api
    calls, per spec section 13) against a test model with a required string,
    an optional/defaulted int, and a nested submodel
    (`TestMakeStrictSchema` in `tests/unit/test_llm_client.py`).
  - cache concurrency invariant documented in place, not changed:
    `clipper/llm/cache.py` has no lock guarding `get_cached_response`'s
    read-then-write-on-global-hit, which is safe under the current fully
    synchronous file i/o because a given key is only ever written on a
    genuine cache miss (no code path overwrites an existing entry with
    different content) -- added a comment on `get_cached_response`
    explaining this and calling out that wrapping the file i/o in
    `asyncio.to_thread` (to stop it blocking the event loop during
    concurrent judge-stage calls) would invalidate the reasoning and require
    a lock like `record_stat`'s.
  - extracted `clipper/util/atomic_write.py` (`atomic_write_json`,
    `_replace_with_retry`) out of the near-verbatim duplication between
    `clipper/util/log.py`'s `record_stat` and `clipper/llm/cache.py`'s
    former `_atomic_write_json`/`_replace_with_retry` -- both now import and
    call the shared helper. pure refactor: no observable behavior or
    interface changed, confirmed by the full existing `test_log.py`/
    `test_llm_cache.py` suites still passing unmodified.

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
