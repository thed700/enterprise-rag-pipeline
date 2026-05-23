# Changelog — AuraRAG

All notable changes are documented here in reverse-chronological order.

---

## [3.6.0] — 2026-05-23

### Bug Fixes

**BUG-AVATAR** (`app/ui.py`)
- `avatar="◈"` is a Unicode geometric shape, not a real emoji. Streamlit's
  `st.chat_message()` only accepts proper emoji codepoints or image URLs.
  Caused `StreamlitAPIException` crashing the entire app on every message.
  Fixed: replaced with `"🤖"` (assistant) and `"🧑"` (user) at all call sites.

**BUG-META-LEAK** (`app/engine/pipeline.py`)
- `stream_query()` yielded raw LLM tokens including the hidden
  `<<META>>{...}<<END_META>>` hallucination-risk block directly into the SSE
  stream. Users saw raw JSON in their chat bubble. Fixed: a sliding buffer
  suppresses tokens while the meta block is being accumulated. A client-side
  strip in `ui.py` provides an additional safety net.

**BUG-SSE-SOURCES** (`app/engine/pipeline.py`, `app/routers/query.py`)
- The SSE streaming endpoint's meta frame contained only `session_id`,
  `token_count`, `provider`, `model`, and `top_k`. The `sources`,
  `pipeline_trace`, `graded_chunks`, and `reflect_loops` fields were never
  emitted, so source cards and pipeline traces never rendered in streaming mode.
  Fixed: `stream_query_with_meta()` generator yields a final `Dict` after
  exhausting the token stream; the router reads it to build a complete meta
  frame.

**BUG-R** (`app/backend/models.py`)
- Anthropic model IDs used versioned date suffixes
  (`claude-opus-4-5-20251101`, etc.) that the Anthropic API does not accept.
  Fixed: updated to the correct short-form model strings
  (`claude-opus-4-6`, `claude-sonnet-4-6`, etc.).

**BUG-UNSTRUCTURED** (`requirements.txt`)
- `unstructured>=0.14.0` was listed as a dependency but is never imported
  anywhere in the codebase. It brings in a ~2 GB dependency tree
  (detectron2, tesseract, etc.) that caused pip timeouts on HF Spaces and
  bloated the Docker image significantly. Removed.

**BUG-SDK-LICENSE** (`.github/workflows/build-artifacts.yml`)
- `sdkmanager` blocked on an interactive Google Android SDK license prompt
  when installing `build-tools;37.0.0`. Since the CI runner is non-interactive,
  the prompt was never answered and the package was silently skipped. Without
  AIDL (part of build-tools), Buildozer could not compile the APK.
  Fixed: pre-write the authoritative Google SDK license hash files before
  Buildozer runs so `sdkmanager` finds them and skips the prompt entirely.

---

## [3.5.0]

### Bug Fixes
- BUG-AK: Grade node updated to use per-document relevance scores (float
  0–1) instead of a binary keep/reject list, enabling finer filtering.
- BUG-AL: `PromptOverrides` fields changed from `str = ""` to
  `Optional[str] = None`; `model_dump(exclude_none=True)` now correctly
  filters unset prompts so the engine defaults are never silently replaced.
- BUG-AM: SSE meta frame now emitted before `[DONE]` so the UI always
  receives pipeline metadata even if the client closes the stream early.
- BUG-AN: `_should_reflect` routing function returns the `END` constant
  (not the string `"END"`) so LangGraph correctly identifies terminal edges.
- BUG-AO: `engine.query()` now guards against being called from inside a
  running event loop and raises a clear error directing callers to `aquery()`.

---

## [3.4.0]

### Features
- Full LangGraph agentic pipeline: Query Rewrite → Hybrid Retrieve →
  Document Grade → Generate → optional Reflect loop.
- True SSE streaming via `graph.astream()` (no bridge thread).
- `GRADE_THRESHOLD`, `REFLECT_ENABLED`, `MAX_REFLECT_LOOPS`,
  `REWRITE_MAX_TOKENS`, `GRADE_MAX_TOKENS` settings.

---

## [3.3.0]

### Bug Fixes
- BUG-Y: `RerankedRetriever` introduced so cross-encoder reranking is
  active inside `ConversationalRetrievalChain` (was bypassed entirely).
- BUG-Z: `arerank()` uses `asyncio.get_running_loop()` (not deprecated
  `get_event_loop()`).
- BUG-AB: `CrossEncoderReranker.shutdown()` added; called from
  `RAGEngine.shutdown()` in FastAPI lifespan cleanup.
- BUG-AC: `_evict_stale()` reads `settings.SESSION_TTL_MINUTES` at call
  time (was reading a stale module-level constant).
- BUG-AE: Source snippet length reads `settings.SOURCE_SNIPPET_LEN`
  (was hardcoded to 300).

---

## [3.2.0]

### Bug Fixes
- BUG-Q: `setup_logging()` respects `LOG_LEVEL` env var.
- BUG-S: `top_k` forwarded correctly from API → engine → reranker.
- BUG-U: `SessionMemoryStore.clear()` removes `_last_access` entry.
- BUG-V: Stream exceptions propagated to SSE error frames.
- BUG-W: `TextLoader` uses UTF-8 + `autodetect_encoding=True`.
- BUG-X: `_seen_hashes` persisted in BM25 pickle; atomic writes via
  `.tmp` rename.
