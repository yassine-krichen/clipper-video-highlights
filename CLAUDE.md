# clipper

source of truth for design and rationale is `clipper-spec.md` at the repo root. read it
before writing any code. this file is a short pointer, kept in sync with spec section 13.

## repo layout (phase 1 subset)

```
clipper/
  pyproject.toml
  clipper/
    cli.py config.py types.py
    llm/            client.py cache.py registry.py
    prompts/        generator_v1.md judge_v1.md
    stages/         probe.py transcribe.py atomize.py generate.py
                    prefilter.py judge.py refine.py gate.py dedupe.py select.py
    generators/     base.py llm_pass.py discourse.py lexical.py coverage.py query.py
    transcribers/   base.py groq.py faster_whisper.py
    embed/          base.py local.py api.py
    media/          ffmpeg.py probe.py
    eval/           metrics.py runner.py
    util/           timing.py log.py hashing.py
  tests/
    fixtures/       sample_3min.mp4, golden/
    unit/
    integration/
  runs/             gitignored, one directory per run
  DECISIONS.md      ambiguities resolved, not settled by the spec
```

## commands

```
uv sync
uv run clipper --help
uv run pytest
uv run ruff check .
```

## working rules (condensed from spec section 13)

- python 3.12, `uv` for deps, `ruff` for lint/format, `pytest`. type hints everywhere,
  pydantic v2 for every serialized structure.
- never mock ffmpeg/ffprobe in tests; use `tests/fixtures/sample_3min.mp4`. mock only
  network calls (llm, transcription) with recorded responses.
- all timestamps are `int` milliseconds. a function taking or returning seconds is a bug.
- the llm never sees or emits raw timestamps, only `[atom_id]` references.
- stages are pure with respect to the run directory: same inputs -> same outputs, apart
  from llm nondeterminism, which the cache removes on re-run.
- no new third-party dependency without a one-line justification in the commit message.
- no queue, no redis, no docker, no web until the phase that names them.
- when something is ambiguous, pick the simplest option consistent with spec section 1,
  log it in `DECISIONS.md`, and continue. only stop if it blocks a phase's "done when".
- keep this file in sync with the spec: when a design decision changes, update the spec
  first, then the code.
- lowercase, no emojis, minimal markdown in generated docs and comments.
