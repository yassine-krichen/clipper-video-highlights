# clipper: long-form video to short-form clips

status: implementation spec, v1
date: 2026-09-05
audience: claude code. read this whole file before writing any code.

## 0. one paragraph

take a long mp4 (podcast, interview, lecture, stream). produce the n best self-contained short clips with exact timestamps, a score breakdown, and a reason. clip discovery is the product. rendering, captions and reframing are downstream and replaceable. the first milestone is not "the app makes vertical videos", it is "given an unseen video, the engine picks the moments a good editor would pick, with natural boundaries, and we can measure that".

## 1. non-negotiable principles

- the engine is a python package with a cli. no web, no queue, no db until phase 3.
- every stage reads and writes plain json files in a run directory. any stage can be re-run alone.
- the llm never outputs timestamps. it selects sentence ids. code maps ids to word-level timings.
- one time unit everywhere: integer milliseconds. field names end in `_ms`.
- one score scale: dimensions are integers 1-10, final score is a float 0-100, computed by code from an explicit formula. never store a bare scalar without its components.
- over-generate candidates, then filter. recall early, precision late.
- every llm call is versioned, cached by content hash, schema-validated, and retried. invalid output never crashes a run.
- every added signal or stage must be justified by the eval harness (section 11). no complexity because a diagram looks smarter.
- read open-source clippers for ideas, never copy their code. podcli is agpl-3.0. see section 14.

## 2. inputs and outputs

### 2.1 input

```
clipper run <video.mp4> [--config config.yaml] [--out runs/<id>/]
```

config keys (yaml, all optional, defaults shown):

```yaml
clip_count: 5
min_duration_ms: 25000
max_duration_ms: 90000
content_type: podcast        # podcast | interview | lecture | stream | auto
language: auto               # iso 639-1 or auto
keywords: []                 # steer candidate generation
editorial_instructions: ""   # free text appended to the judge prompt
platform: shorts             # shorts | tiktok | reels | generic
allow_multi_segment: false
transcriber: groq            # groq | faster_whisper
llm_generator: gemini-flash  # model alias, see section 9
llm_judge: gemini-flash
embedder: local              # local | api
render: false                # phase 2+
captions: false              # phase 3+
reframe: none                # none | center | face   (phase 4+)
```

### 2.2 run directory

every run writes to `runs/<run_id>/` with this exact layout. later stages consume earlier files. re-running a stage overwrites only its own outputs.

```
runs/<run_id>/
  input.json          probe result + resolved config
  audio.wav           16khz mono pcm extracted by ffmpeg
  transcript.json     Transcript
  atoms.json          Atom[] (sentences with word timings)
  features.json       audio features per 500ms bucket   (phase 2+)
  candidates.json     Candidate[] raw pool with generator provenance
  judged.json         JudgedCandidate[] after llm judge
  refined.json        RefinedCandidate[] after boundary refinement
  selected.json       ClipManifest final selection
  clips/              rendered mp4s                      (phase 2+)
  llm_cache/          one json per cached call
  log.jsonl           structured log, one event per line
  stats.json          counts per stage, cost, wall time
```

### 2.3 clip manifest

```json
{
  "run_id": "2026-09-05T14-02-11_ep17",
  "source": {"path": "episode_17.mp4", "duration_ms": 3612400, "sha256": "..."},
  "config": {"...": "resolved config"},
  "clips": [
    {
      "id": "clip_01",
      "segments": [{"start_ms": 2477200, "end_ms": 2512400}],
      "duration_ms": 35200,
      "atom_range": [412, 419],
      "score": 84.6,
      "dimensions": {
        "hook": 9, "value": 8, "emotion": 7, "curiosity": 9,
        "novelty": 8, "completeness": 9, "payoff": 8
      },
      "standalone_level": 0,
      "penalties": {"boundary": 0.0, "silence": 1.2, "redundancy": 0.0},
      "reason": "opens on a concrete mistake, explains the cause, ends on the lesson.",
      "title": "the hiring mistake that cost us six months",
      "transcript": "the biggest mistake i made...",
      "generators": ["llm_pass", "discourse"],
      "speaker": null
    }
  ],
  "stats": {"...": "see stats.json"}
}
```

`segments` is always a list. one element in the contiguous case. this is the only place multi-segment is represented.

## 3. core data types

define these as pydantic v2 models in `clipper/types.py`. everything on disk is one of these serialized.

```python
class Word(BaseModel):
    text: str
    start_ms: int
    end_ms: int
    confidence: float | None = None

class Atom(BaseModel):
    """a sentence-ish unit. the smallest thing the llm may reference."""
    id: int                      # 0-based, contiguous, stable within a run
    start_ms: int
    end_ms: int
    text: str
    words: list[Word]
    speaker: str | None = None
    pause_before_ms: int         # silence gap to previous atom, 0 for first
    pause_after_ms: int

class Transcript(BaseModel):
    language: str
    duration_ms: int
    words: list[Word]
    segments: list[dict]         # raw provider segments, kept for debugging
    provider: str
    model: str

class Candidate(BaseModel):
    id: str
    atom_start: int              # inclusive
    atom_end: int                # inclusive
    start_ms: int
    end_ms: int
    generators: list[str]        # which generators proposed this (merged)
    generator_hint: str | None   # free text from llm_pass, e.g. "a story about hiring"
    pre_score: float             # 0-1, deterministic

class JudgeResult(BaseModel):
    candidate_id: str
    dimensions: dict[str, int]   # keys fixed: hook value emotion curiosity novelty completeness payoff, each 1-10
    standalone_level: int        # 0 fully standalone, 1 minor gap, 2 imperfect, 3 depends on prior context
    suggested_atom_start: int
    suggested_atom_end: int
    reason: str                  # <= 40 words
    title: str                   # <= 12 words, must be grounded in transcript
    missing_context: str | None  # what the viewer would not understand

class JudgedCandidate(Candidate):
    judge: JudgeResult
    score: float                 # 0-100 computed by code, see section 8

class RefinedCandidate(JudgedCandidate):
    segments: list[Segment]
    refinement_log: list[str]
    penalties: dict[str, float]
    embedding: list[float] | None
```

## 4. pipeline

```
probe -> extract audio -> transcribe -> atomize -> [features] -> generate candidates
  -> prefilter -> judge -> refine boundaries -> gate -> dedupe -> select -> [render]
```

each stage is a function `run_<stage>(run_dir: Path, cfg: Config) -> None` in `clipper/stages/<stage>.py`. it reads its inputs from the run dir, writes its outputs, appends to `log.jsonl` and `stats.json`. `clipper run` calls them in order. `clipper stage <name> <run_dir>` runs one.

### 4.1 probe and extract

- `ffprobe -v quiet -print_format json -show_format -show_streams`. never trust the filename or extension.
- reject if: no audio stream, duration < 60s, duration > 4h (configurable), unreadable.
- extract audio: `ffmpeg -i in.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le audio.wav`.
- write `input.json` with probe result, sha256 of the file, resolved config.

### 4.2 transcribe

interface:

```python
class Transcriber(Protocol):
    def transcribe(self, wav: Path, language: str | None) -> Transcript: ...
```

default `GroqTranscriber`: model `whisper-large-v3-turbo`, `response_format=verbose_json`, `timestamp_granularities=["word","segment"]`. groq caps request size, so chunk the wav into 10-minute pieces with 2s overlap, transcribe each, stitch by dropping words in the overlap zone that already appear in the previous chunk (match on timing, not text). cost is about $0.04 per hour of audio, no gpu.

fallback `FasterWhisperTranscriber`: `faster-whisper`, model `large-v3-turbo` if gpu else `small`, `word_timestamps=True`, `vad_filter=True`. only used when `transcriber: faster_whisper` or when groq returns a non-retryable error.

language: pass through if set, otherwise take the provider's detection and store it in `Transcript.language`.

### 4.3 atomize

turn the flat word list into `Atom[]`. deterministic, no llm.

- split on sentence-final punctuation (`. ? !`) from the provider text, and additionally on any pause >= 700ms between words, and on speaker change when diarization exists.
- merge atoms shorter than 3 words into the following atom unless a pause >= 700ms separates them.
- an atom longer than 25s is split at the longest internal pause.
- compute `pause_before_ms` / `pause_after_ms` from neighbouring word timings.
- ids are 0..n-1 in order. every later stage references atoms by id.

test: total atom span equals transcript span; no gaps in ids; every word belongs to exactly one atom.

### 4.4 features (phase 2)

per 500ms bucket from `audio.wav` using numpy + librosa: rms, rms delta vs trailing 5s mean, speech ratio (from transcript coverage), silence flag. later: laughter probability via yamnet onnx (what podcli uses; no torch dependency). write `features.json`. audio energy is a candidate signal, never a quality signal.

### 4.5 generate candidates

generators live in `clipper/generators/`. each returns `list[Candidate]` with `generators=[its_name]`. the stage merges them, then normalizes.

**g1 `llm_pass` (primary).** one call. input: every atom as `[id] speaker: text`, plus content_type, keywords, editorial_instructions, duration bounds. a 60-minute episode is roughly 12k tokens, so this is cheap on a long-context flash model. ask for 20-40 proposals, each `{atom_start, atom_end, hint}`. this is the best single generator because the model sees the whole arc. if the transcript exceeds the model's practical limit (~150k tokens), split into 45-minute chunks with 5-minute overlap and merge.

**g2 `discourse`.** deterministic. walk atoms; start a new region at any `pause_before_ms >= 1500` or speaker change; emit every contiguous atom run inside a region whose span fits the duration bounds, capping at 4 candidates per region (longest, shortest-valid, and two intermediate).

**g3 `lexical`.** deterministic. atoms containing hook patterns (`clipper/generators/hooks.yaml`: "biggest mistake", "nobody tells you", "the truth is", "i lost", "here's what happened", numbers with currency, etc. plus a per-language list). emit the window from that atom forward to the first atom whose end satisfies min_duration, and one extended to max_duration.

**g4 `coverage`.** deterministic safety net. after g1-g3, find any atom not inside any candidate; emit a min_duration window starting at it. this guarantees every part of the video is at least considered.

**g5 `query`.** only when keywords are set. embed atoms, cosine against keyword embeddings, emit windows around top-20 atoms.

**g6 `audio_event`** (phase 2). windows around rms spikes and laughter.

**g7 `native_video`** (phase 5, experimental). upload audio (or low-res video) directly to gemini and ask for moments with mm:ss timestamps; snap to nearest atoms. useful for gaming and demos where the transcript misses the highlight.

normalization: snap all candidates to atom boundaries; drop those outside duration bounds; merge candidates whose atom ranges are identical, unioning `generators`. write `candidates.json` and per-generator counts to `stats.json`.

### 4.6 prefilter

deterministic, cheap, only to bound judge calls. `pre_score` in 0-1:

```
pre_score = 0.40 * (1 if "llm_pass" in generators else 0)
          + 0.20 * duration_fit          # 1 at the middle of the range, 0 at the edges
          + 0.15 * min(1, n_generators / 3)
          + 0.15 * (1 - silence_ratio)
          + 0.10 * lexical_hit
```

keep top `judge_cap` (default 40). always keep every `llm_pass` candidate even if that exceeds the cap. this stage exists to bound cost, not to be clever.

### 4.7 judge

one llm call per candidate, run concurrently (default 8 in flight). input: `previous context` (atoms up to 60s before), `candidate` (its atoms, each prefixed with id), `following context` (up to 30s after), content_type, platform, editorial_instructions. output `JudgeResult` via structured output / json schema. the prompt is `clipper/prompts/judge_v1.md`. its core instruction:

> you are an experienced short-form editor. judge whether the candidate region can be a standalone clip for a viewer who has not seen the source. score hook, value, emotion, curiosity, novelty, completeness, payoff from 1 to 10. rate standalone_level 0-3. do not reward statements that merely sound important. prefer complete thoughts and clear closure. suggest the best start and end atom ids within or adjacent to the candidate (at most 2 atoms outside it on either side). the title must describe what is literally said, not an invented claim.

after the call, code computes `score` (section 8). write `judged.json`.

### 4.8 refine boundaries

deterministic. input is the judge's suggested atom range.

- start: atom start of `suggested_atom_start`, minus `min(300, pause_before_ms / 2)` ms of lead-in so the first word is not clipped.
- end: atom end of `suggested_atom_end`, plus `min(400, pause_after_ms / 2)` ms.
- if the result violates duration bounds, shrink or grow one atom at a time from the end that changes the judge's completeness least: prefer dropping a trailing atom over a leading one, prefer adding a following atom over a preceding one. log every step to `refinement_log`.
- hard rule: a boundary never lands inside a word.
- multi-segment (only when `allow_multi_segment`): if an internal atom run of 3-15s has silence_ratio > 0.6 or matches a filler pattern, propose removing it; keep only if the judge, re-asked with the cut transcript, returns completeness >= the original. otherwise stay contiguous.

compute penalties: `boundary` (0 unless the start or end atom begins with a dependent word like "so", "because", "and", "that's why": +5 each), `silence` (5 * silence_ratio of the clip), `redundancy` set later by dedupe.

### 4.9 gate

hard rejects: standalone_level == 3; duration outside bounds after refinement; score < 40; transcript confidence mean < 0.5 with fewer than 15 words. log reason for every reject.

### 4.10 dedupe

- temporal: interval iou > 0.35 -> keep the higher score, mark the other rejected with reason.
- semantic: embed each clip transcript (`local`: `bge-small-en-v1.5` via sentence-transformers; `api`: any openai-compatible embeddings endpoint). cosine > 0.85 -> keep the higher score. the loser's score is recorded with `penalties.redundancy` so it can be inspected.

### 4.11 select

greedy diversity-aware selection, `clip_count` rounds:

```
pick = argmax over remaining of
    score
  - 25 * max_cosine(candidate, already_selected)
  - 10 * timeline_proximity(candidate, already_selected)   # 1 if within 120s of a pick, decays to 0 at 10 min
```

soft, so a section containing the five best moments can still yield multiple clips. write `selected.json` as a `ClipManifest`. also write a human-readable `selected.md` with timestamps in mm:ss, title, score, reason, transcript.

### 4.12 render (phase 2)

separate command: `clipper render runs/<id>/ [--clip clip_01] [--vertical] [--captions]`. re-rendering never re-runs discovery.

- accurate cut: `ffmpeg -ss <start> -i in.mp4 -t <dur> -c:v libx264 -preset veryfast -crf 20 -c:a aac -b:a 160k -movflags +faststart`. use `-ss` before `-i` for speed only when the source has frequent keyframes; validate boundary accuracy in tests against a fixture with a known tone at a known time.
- multi-segment: cut each segment then concat with the concat demuxer.
- loudness: `-af loudnorm=I=-16:TP=-1.5:LRA=11`.
- captions (phase 3): generate ass from atom words; 1-2 lines, max 22 chars per line, group by phrase, karaoke highlight optional; burn in with `subtitles=` filter. caption styling never touches the score.
- reframe (phase 4): `center` is a static 9:16 center crop. `face`: opencv yunet face detection at 5 fps, exponential smoothing + deadzone + velocity cap on the crop x, single virtual camera path, `crop` filter with expression driven by a keyframe file. split-screen sources: detect two stable face clusters and use a stacked layout.

## 5. cli

```
clipper run <video> [--config] [--out] [--stop-after <stage>]
clipper stage <stage> <run_dir>
clipper render <run_dir> [--clip id] [--vertical] [--captions]
clipper inspect <run_dir>            # prints candidates table, per-stage counts, rejects with reasons, cost
clipper eval <golden_dir> [--config] # section 11
clipper cost <run_dir>
```

built with `typer`. output to stdout is human-readable; `--json` flag for machine output.

## 6. repository layout

```
clipper/
  pyproject.toml            # uv-managed, python 3.12
  README.md
  CLAUDE.md                 # working rules, section 13, kept in sync with this spec
  clipper/
    __init__.py
    cli.py
    config.py               # pydantic-settings, yaml + env
    types.py
    llm/
      client.py             # openai-compatible client, structured output, retry, cache
      cache.py
      registry.py           # model aliases -> provider/model/base_url
    prompts/
      generator_v1.md
      judge_v1.md
    stages/
      probe.py transcribe.py atomize.py features.py generate.py
      prefilter.py judge.py refine.py gate.py dedupe.py select.py render.py
    generators/
      base.py llm_pass.py discourse.py lexical.py coverage.py query.py audio_event.py
      hooks.yaml
    transcribers/
      base.py groq.py faster_whisper.py
    embed/
      base.py local.py api.py
    media/
      ffmpeg.py probe.py captions.py reframe.py
    eval/
      metrics.py runner.py
    util/
      timing.py log.py hashing.py
  tests/
    fixtures/
      sample_3min.mp4       # short real recording, committed, < 15mb
      sample_3min.transcript.json
      golden/               # section 11 format
    unit/
    integration/
  runs/                     # gitignored
```

## 7. llm layer

- all calls go through `clipper/llm/client.py`, which speaks the openai chat completions format with `response_format` json schema. base url and key come from config. this makes gemini, groq, openai, anthropic (via a compatible proxy) and a self-hosted proxy interchangeable.
- model aliases in `registry.py`: `gemini-flash`, `gemini-pro`, `groq-llama`, `claude-sonnet`. config uses aliases; the registry resolves them. defaults: generator and judge both `gemini-flash`. document in the readme how to point aliases at a self-hosted openai-compatible proxy.
- every call is cached at `runs/<id>/llm_cache/<sha256(model + prompt_version + input)>.json` and also in a global `~/.clipper/cache/`. re-running a stage with unchanged inputs costs nothing.
- every prompt file has a version in its name. the version is part of the cache key and is written into `stats.json`.
- invalid json or schema failure: retry up to 3 times with the validation error appended. after that, the candidate gets `dimensions` all 5, `standalone_level` 2, reason "judge failed", and a log event. never crash.
- record tokens in/out and estimated cost per call. `clipper cost` sums them.
- temperature 0 for judge, 0.3 for generator.

## 8. scoring

```
base = 0.22*hook + 0.16*value + 0.10*emotion + 0.12*curiosity
     + 0.10*novelty + 0.18*completeness + 0.12*payoff        # each 1-10, base in 1-10

standalone_adj = {0: 0, 1: -4, 2: -12, 3: reject}[standalone_level]

score = clamp(base * 10 + standalone_adj
              - penalties.boundary - penalties.silence - penalties.redundancy
              + keyword_bonus, 0, 100)                       # keyword_bonus max 5
```

weights live in `config.py` under `scoring.weights` and can be overridden per content_type (`lecture` raises value and completeness, `stream` raises emotion and lowers completeness). weights are heuristics to be tuned against the eval harness, not truths.

## 9. content-type profiles

a profile adjusts: duration defaults, scoring weights, hook list, generator prompt preamble. shipped profiles: `podcast` (default), `interview`, `lecture`, `stream`. `auto` asks the generator model to classify from the first 5 minutes of transcript and picks one. profiles are yaml in `clipper/profiles/`.

## 10. observability

`log.jsonl` events: `{ts, stage, event, data}`. `stats.json`:

```json
{
  "stages": {"transcribe": {"wall_ms": 41200, "cost_usd": 0.04}, "judge": {"wall_ms": 38000, "calls": 40, "cost_usd": 0.11}},
  "candidates": {"llm_pass": 31, "discourse": 212, "lexical": 18, "coverage": 4, "merged": 240, "after_prefilter": 44, "judged": 44, "after_gate": 29, "after_dedupe": 17, "selected": 5},
  "models": {"transcriber": "groq/whisper-large-v3-turbo", "generator": "gemini-flash@generator_v1", "judge": "gemini-flash@judge_v1"},
  "total_cost_usd": 0.16,
  "total_wall_ms": 131000
}
```

`clipper inspect` renders this plus a table of every candidate with its fate. this is the primary debugging tool. build it in phase 1, not later.

## 11. evaluation harness

this is how every decision after phase 1 gets made.

### 11.1 golden set format

`tests/fixtures/golden/<name>/`:

```
video.mp4                   or video.url (yt-dlp) if too big to commit
transcript.json             cached, so eval never pays for transcription twice
annotations.json
```

```json
{
  "annotator": "yassine",
  "content_type": "podcast",
  "moments": [
    {"start_ms": 2477200, "end_ms": 2512400, "tier": 1, "note": "the hiring story"},
    {"start_ms": 901000,  "end_ms": 958000,  "tier": 2, "note": "funny tangent"}
  ],
  "traps": [
    {"start_ms": 1400000, "end_ms": 1440000, "why": "sounds important, depends on earlier context"}
  ]
}
```

tier 1 = must find, tier 2 = good to find. traps = must not select. start with 5 videos across podcast, interview, lecture. grow to 15.

### 11.2 metrics

`clipper eval` runs the pipeline on every golden video with the given config and reports:

- `hit@n`: fraction of tier-1 moments with iou >= 0.5 against any selected clip.
- `recall_pool`: fraction of tier-1 moments with iou >= 0.5 against any candidate in the pool before judging. if this is low the generators are the problem, not the judge.
- `trap_rate`: fraction of traps that got selected.
- `boundary_mae_ms`: mean abs error of start and end vs the annotation, for hits.
- `dup_rate`: pairs of selected clips with semantic cosine > 0.85.
- `spread`: std dev of selected clip midpoints over the source duration.
- `cost_per_hour_usd`, `wall_per_hour_s`.

writes `eval/<timestamp>_<config_name>.json` and prints a table. keep every result; comparing two config files is the ablation study.

### 11.3 targets for the end of phase 1

hit@5 >= 0.6 on tier-1, recall_pool >= 0.9, trap_rate <= 0.1, boundary_mae <= 2500ms, cost <= $0.50 per source hour.

## 12. phases

each phase ends with a git tag. do not start the next phase until "done when" holds.

### phase 1: extraction engine, cli only

in scope: sections 3, 4.1-4.3, 4.5 (g1-g4), 4.6, 4.7, 4.8 contiguous only, 4.9-4.11, 5 (`run`, `stage`, `inspect`, `cost`), 6, 7, 8, 10, 11.
out of scope: rendering, features, diarization, web, queue, db, multi-segment, reframing, captions.
done when: `clipper run tests/fixtures/sample_3min.mp4` produces a valid manifest; `clipper eval` runs on 5 golden videos and prints metrics; unit tests for atomize, prefilter, refine, dedupe, select pass; `clipper inspect` shows every candidate's fate; total cost on a 60-minute video is under $0.50.

### phase 2: render and audio features

in scope: 4.12 contiguous cut + loudnorm; 4.4 rms/silence; g6; `silence` penalty using real features; a render accuracy test (fixture with a beep at a known ms, assert the beep sits within 80ms of the expected offset in the output).
done when: `clipper render` produces playable clips for every selected clip; eval metrics do not regress.

### phase 3: captions, multi-segment, service layer

in scope: ass captions burned in; `allow_multi_segment`; fastapi service in `apps/api/` wrapping the cli stages as jobs; jobs table in postgres (`select ... for update skip locked` worker loop, no redis until proven needed); object storage for inputs and outputs (s3-compatible); endpoints: `post /videos` (signed upload url), `post /videos/{id}/runs`, `get /runs/{id}`, `get /runs/{id}/clips`, `post /clips/{id}/render`, `post /clips/{id}/review` (approve/reject/edit with reason, stored for the feedback dataset).
done when: an upload via the api produces a manifest and rendered clips, and a worker crash mid-run resumes from the last completed stage on restart.

### phase 4: review ui and reframing

in scope: next.js app in `apps/web/`: upload, stage progress, results with preview, timeline showing every candidate as a bar colored by fate, approve/reject/adjust in/out by dragging on the transcript, re-render; `reframe: center` and `reframe: face`; optional diarization behind `Diarizer` protocol (pyannote community-1 locally, or assemblyai/deepgram diarization via the transcriber interface) feeding `Atom.speaker` and speaker-aware discourse boundaries.
done when: a non-technical user can go upload -> review -> download vertical captioned clips without touching the cli.

### phase 5: learning loop and experiments

in scope: g7 native video generator; per-project knowledge file (show description, recurring segments, audience, past clip titles) injected into generator and judge prompts; review decisions exported as training data; a small learned prefilter (logistic regression on pre_score features + judge dimensions) trained on reviews, evaluated against the heuristic on the golden set before it is allowed to replace it.
done when: at least one experiment shows a measured improvement on `clipper eval`, and the change ships behind a config flag.

## 13. working rules for claude code

- python 3.12, `uv` for deps, `ruff` for lint and format, `pytest`. type hints everywhere, pydantic v2 for every serialized structure.
- never mock ffmpeg or ffprobe in tests. use `tests/fixtures/sample_3min.mp4`. mock only network calls (llm, groq) with recorded responses in `tests/fixtures/recorded/`.
- all timestamps are `int` milliseconds. any function that takes or returns seconds is a bug.
- the llm never sees or emits raw timestamps. it sees `[atom_id]` prefixes and emits atom ids.
- stages are pure with respect to the run directory: same inputs, same outputs, apart from llm nondeterminism, which the cache removes on re-run.
- no new third-party dependency without a one-line justification in the commit message.
- no queue, no redis, no docker, no web until the phase that names them. the engine must run with `uv run clipper run video.mp4` on a laptop with only ffmpeg installed and api keys in `.env`.
- keep `CLAUDE.md` in sync with this spec: repo layout, commands, rules. when a design decision changes, update this spec first, then the code.
- commit at the end of every stage implementation, tag at the end of every phase.
- when something is ambiguous, pick the simplest option consistent with section 1, note it in `DECISIONS.md`, and continue. do not stop to ask unless the ambiguity blocks a phase's "done when".
- lowercase, no emojis, minimal markdown in all generated docs and comments.

## 14. references and licenses

read for ideas. do not copy code from agpl repositories into this mit/proprietary codebase.

- podcli (github.com/nmbrthirteen/podcli), agpl-3.0. best reference for face tracking with yunet, caption styles, laughter detection via yamnet onnx, knowledge-base prompting, mcp exposure. ideas only.
- ai-highlight-clip (github.com/toki-plus/ai-highlight-clip). whisper + sliding windows + llm scoring; the baseline this design must beat.
- whisperx (bsd-2), pyannote.audio (mit, models gated behind an hf token), faster-whisper (mit), sentence-transformers (apache-2).
- rhapsody dataset (arxiv 2505.19429), podcast highlight labels derived from youtube most-replayed; candidate source for a larger eval set in phase 5.
- gemini video understanding docs (ai.google.dev/gemini-api/docs/video-understanding) for g7: 1 fps sampling, mm:ss timestamps, low media resolution for long inputs.
- commercial landscape: opus clip (multimodal, 0-100 virality score, genre models), vizard (transcript-first, volume pricing), choppity (custom ai criteria). users complain about credits burning on upload minutes and weak performance on unstructured conversation. this engine competes on transparent per-dimension scores, custom editorial instructions, and honest standalone gating, not on a single virality number.

## 15. decisions log

`DECISIONS.md` in the repo root, one dated line per decision claude code makes that this spec did not settle. reviewed at each phase boundary.
