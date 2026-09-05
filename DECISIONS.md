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
