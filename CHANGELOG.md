# Changelog

All notable changes to AuraRAG are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [3.3] — 2026-05-17

### Fixed

| ID | File(s) | Severity | Description |
|----|---------|----------|-------------|
| BUG-Y | `engine.py` | 🔴 High | `CrossEncoderReranker` was initialized and documented but completely bypassed in `query()` and `stream_query()` — both used `_build_hybrid_retriever()` directly, feeding raw hybrid results to the LLM. The reranker only ran in the standalone `retrieve()` method, which no API endpoint calls. Fixed: introduced `RerankedRetriever`, a `BaseRetriever` subclass wrapping hybrid search + cross-encoder reranking; `_build_reranked_retriever(top_k)` now used everywhere a retriever is passed to `ConversationalRetrievalChain`. |
| BUG-Z | `engine.py` | 🟡 Medium | `CrossEncoderReranker.arerank()` called the deprecated `asyncio.get_event_loop()` inside a running event loop — `DeprecationWarning` on Python 3.10+, will `raise RuntimeError` in a future release. Fixed: replaced with `asyncio.get_running_loop()`. |
| BUG-AA | `app/ui.py` | 🟡 Medium | `_api_stream()` sent no `top_k` field to `/query/stream` — every streaming query silently fell back to the server-side default of 5 regardless of the caller's preference. Fixed: `top_k` now included in the streaming POST payload, consistent with `_api_query()`. |
| BUG-AB | `engine.py` · `main.py` | 🟡 Medium | `CrossEncoderReranker._executor` (a `ThreadPoolExecutor`) was never shut down, leaking OS threads on every hot-reload or graceful shutdown. Fixed: added `CrossEncoderReranker.shutdown()` and `RAGEngine.shutdown()`, called from the FastAPI lifespan cleanup block. |
| BUG-AC | `engine.py` | 🟡 Medium | `SessionMemoryStore._evict_stale()` read the module-level constant `SESSION_TTL_MINUTES = 60` instead of `settings.SESSION_TTL_MINUTES`. Setting the env var in `.env` had zero effect at runtime. Fixed: `_evict_stale()` now calls `get_settings().SESSION_TTL_MINUTES`. |
| BUG-AD | `models.py` | 🟡 Medium | `HealthResponse` was missing the `bm25_docs` field that `engine.health()` returns and the v3.2.0 changelog documented. FastAPI's response serialiser silently dropped the field from every `GET /health` response. Fixed: added `bm25_docs: str = "0"` to `HealthResponse`. |
| BUG-AE | `engine.py` · `utils.py` | 🟢 Low | Source snippet truncation in `query()` was hardcoded to `[:300]` chars. Fixed: added `SOURCE_SNIPPET_LEN: int = 300` to `Settings` so operators can tune via `.env` without code changes. |

### Tests
- 9 new regression tests covering BUG-Y (retriever type in query + stream), BUG-Z (source inspection), BUG-AA (payload inspection), BUG-AB (shutdown delegation), BUG-AC (settings read), BUG-AD (model field + serialisation), BUG-AE (snippet length).
- All v3.2.0 tests retained and passing.

---

## [3.2.0] — 2026-05-16

### Fixed

| ID | File(s) | Severity | Description |
|----|---------|----------|-------------|
| BUG-S | `models.py` · `main.py` · `engine.py` | 🔴 High | `top_k` declared on `QueryRequest` / `StreamQueryRequest` but never forwarded to the engine — reranker and source-slice always defaulted to 5. Threaded through the full API → engine → reranker call chain. `StreamQueryRequest` was also missing the field entirely. |
| BUG-V | `engine.py` | 🔴 High | `stream_query()` created a background chain task and consumed the callback iterator, but exceptions raised by the chain were silently swallowed — the async generator hung indefinitely instead of emitting an error frame. Chain exceptions are now re-raised into the generator and surfaced as `{"error": ...}` SSE frames. |
| BUG-X | `engine.py` | 🔴 High | `_seen_hashes` was rebuilt from `_all_docs` on BM25 pickle restore. Legacy v3.0.0 pickles stored only a bare `BM25Retriever` with no docs, so `_seen_hashes` came back empty after an upgrade — defeating dedup. Fixed: hashes are now stored explicitly in the pickle under `"hashes"`. Pickle writes are also atomic (`write → .tmp`, then `os.replace`) to prevent corruption on unclean shutdown. |
| BUG-W | `main.py` · `Dockerfile` · `requirements.txt` | 🔴 High | `TextLoader(path)` used the OS locale encoding. In C-locale Docker containers any non-ASCII `.txt` file raised `UnicodeDecodeError`. Fixed: `TextLoader(path, encoding="utf-8", autodetect_encoding=True)`. Added `python-magic` to `requirements.txt` and `libmagic1` to both Dockerfile stages. |
| BUG-U | `engine.py` | 🟡 Medium | `SessionMemoryStore.clear()` cleared the memory object but left its entry in `_last_access`, so the session was still counted as active and a fresh `get()` call could not create a clean replacement until TTL eviction. Fixed: `clear()` now removes from both `_sessions` and `_last_access`. |
| BUG-Q | `utils.py` · `main.py` | 🟡 Medium | `setup_logging()` always logged at `INFO` regardless of `Settings.LOG_LEVEL`. The call site in `main.py` passed no argument, silently ignoring the env var. Fixed: `setup_logging()` reads `get_settings().LOG_LEVEL` internally; `basicConfig` uses `force=True` to override uvicorn's earlier handler. |
| BUG-R | `constants.py` | 🟡 Medium | `claude-opus-4-5` and `claude-sonnet-4-5` are not valid Anthropic API model IDs — calls with these strings would fail at runtime. Corrected to `claude-opus-4-6` and `claude-sonnet-4-6`. Added `o1-mini` / `o1-preview` to the OpenAI list. |
| BUG-T | `models.py` | 🟢 Low | `IngestResponse.message` typed `Optional[str]` but always populated at the call site. Changed to `str = ""` to match actual behaviour and eliminate unnecessary null-checks. |

### Changed
- `HealthResponse` now includes a `bm25_docs` field alongside `docs_indexed` so monitoring can independently verify BM25 and ChromaDB corpus sizes.
- `docker-compose.yml` documents the Redis service wiring path (still commented out; activate by uncommenting and setting `REDIS_URL`).
- `.env.example` updated to v3.2.0 with `REDIS_URL` documentation.
- `Dockerfile` bumped `LABEL org.opencontainers.image.version` to `3.2.0`.

### Tests
- 9 new regression tests covering BUG-S, BUG-U, BUG-V (structure), BUG-X, BUG-Q, BUG-R, BUG-T, and the new `bm25_docs` health field.
- All 3.1.0 tests retained and passing.

---

## [3.1.0] — 2025-04-XX

### Fixed
16 bugs — see v3.1.0 README for the full table (BUG-A through BUG-P, BUG-10).

### Added
- `GET /providers` endpoint — UI fetches provider list over HTTP instead of importing `app.engine`.
- `POST /query/stream` — true SSE streaming via `AsyncIteratorCallbackHandler`.
- `DELETE /memory/{session_id}` and `DELETE /memory` — session management endpoints.
- Per-session `ConversationBufferWindowMemory` with TTL eviction (`SessionMemoryStore`).
- Cross-encoder re-ranking (`CrossEncoderReranker`) with async `arerank()`.
- SHA-256 content-hash deduplication on ingest.
- slowapi rate limiting on `/query` (30/min) and `/ingest` (10/min).
- Chunked 256 KB streaming upload to temp file (avoids full-file RAM load).
- `MAX_UPLOAD_MB` enforced during streaming.
- `CHUNK_SIZE` / `CHUNK_OVERLAP` / `SESSION_TTL_MINUTES` configurable from `.env`.

---

## [3.0.0] — initial v3 release

- Multi-provider LLM backend (OpenAI, Anthropic, Google Gemini, Ollama).
- ChromaDB persistent vector store + BM25 hybrid retrieval.
- Streamlit "Bring Your Own Key" UI.
- FastAPI backend with `/ingest`, `/query`, `/health`.
