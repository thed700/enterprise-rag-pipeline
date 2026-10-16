# Changelog — AuraRAG

All notable changes are documented here in reverse-chronological order.

---

## [3.7.0] — 2026-10-15

Stability release. No API-breaking changes; one additive SSE frame (`{"reset": true}`).

### Bug Fixes

**BUG-CTX-TRUNC** (`app/engine/pipeline.py`)
- The generate node truncated every chunk to `SOURCE_SNIPPET_LEN` (300, a UI
  display setting) before sending it to the LLM, so the model saw ~60 % of each
  512-char chunk. `_MAX_CONTEXT_CHARS_PER_DOC` (1200) was defined but never used.
  Fixed: LLM context uses the full chunk; `SOURCE_SNIPPET_LEN` only affects source cards.

**BUG-GRADE-TOKENS** (`app/utils.py`, `.env.example`)
- `GRADE_MAX_TOKENS=64` truncated the grader's JSON (a score per snippet, up to
  10, plus a reason). Parsing failed and grading silently degraded to keyword
  matching on every query. Default raised to 256.

**BUG-STREAM-LEAK** (`app/engine/pipeline.py`)
- The streaming meta filter yielded `<<`, `META` and the closing `>>` fragments
  whenever the marker was split across tokens. Replaced with an incremental
  `_MetaStripper` that holds back partial markers.

**BUG-STREAM-REFLECT** (`pipeline.py`, `routers/query.py`, `ui.py`)
- When the reflect loop re-ran generate, both answers were streamed and the
  concatenation was saved to session history. The engine now emits a reset event,
  the router forwards `{"reset": true}`, the UI clears the bubble, and only the
  final answer is persisted.

**BUG-META-UNCLOSED** (`pipeline.py`, `ui.py`)
- A truncated answer containing `<<META>>` with no closing tag leaked raw JSON.

**BUG-SYNC-PROMPTS** (`pipeline.py`)
- `RAGEngine.query()` silently dropped `system_prompts`.

**BUG-HASH-EARLY** (`pipeline.py`)
- Chunk hashes were recorded before the vector write succeeded; a failed ingest
  permanently blocked re-ingesting that content.

**BUG-HASH-SOURCE** (`pipeline.py`)
- Dedup keyed on text only, so identical text in a second file was dropped and
  never cited under that file. Key is now `source + text`. Hashes are recomputed
  from stored docs on load, so existing indexes upgrade cleanly.

**BUG-INGEST-RACE** (`pipeline.py`)
- Concurrent `/ingest` calls raced on `_all_docs` and the BM25 rebuild. Now locked.

**BUG-BM25-MISSING** (`pipeline.py`)
- If Chroma held data but `bm25.pkl` was lost or corrupt (common on ephemeral
  HF Spaces storage), every query failed. The BM25 index is rebuilt from Chroma.

**BUG-CORS-ENV** (`app/main.py`)
- CORS origins were read from `os.environ`, ignoring `ALLOWED_ORIGINS` in `.env`.

**BUG-INGEST-BLOCKING / BUG-INGEST-500 / BUG-TMP-LEAK** (`app/main.py`)
- Parsing ran on the event loop; unreadable files returned 500; aborted uploads
  leaked temp files. Now: worker thread, clean 422, guaranteed cleanup.

**BUG-LIMITER** (`app/limiter.py`, `main.py`, `routers/query.py`)
- Two unrelated `Limiter` instances; now one shared object.

**BUG-NAN-TEXT / BUG-CSV-ENCODING / BUG-EMPTY-PDF** (`app/backend/ingest.py`)
- `pd.NA`/`NaT` rendered as literal text in every row; non-UTF-8 CSVs crashed;
  scanned PDFs "succeeded" with zero searchable content.

**BUG-LC-UNPINNED / BUG-DEPS-MISSING** (`requirements.txt`)
- LangChain packages were uncapped; langchain 1.x removed the imports this code
  uses. Capped `<1.0`. Added missing `chardet` and `xlrd`; removed unused
  `python-magic` and `redis`.

**BUG-TESTS-STALE** (`tests/test_engine.py`)
- 25 of 47 tests failed: they patched `app.engine.CrossEncoder` etc., which
  stopped existing when the engine moved into `app/engine/pipeline.py` in v3.4.
  Patch targets repaired; new regression tests added for the fixes above.

### Housekeeping
- Removed dead `app/engine.py` (shadowed by the `app/engine/` package) and
  `app/engine_core.py` (legacy v3.3 code, never imported).
- Model registry: added Claude 5.5 family; removed retired Gemini 1.5 models.
- Version strings bumped to 3.7 everywhere. Roadmap: Graph-RAG moved to v3.8.

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
