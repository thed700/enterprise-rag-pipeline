# AuraRAG v3.3 — Release Notes

**Date:** 2026-05-17  
**Tag:** `v3.3`  
**Type:** Patch release (bug fixes only — no breaking changes)

---

## Summary

7 bugs fixed across the engine, models, and UI layers. No API contract changes —
v3.3 is a drop-in upgrade from v3.2.0.

---

## Critical fix (upgrade strongly recommended)

**Cross-encoder reranker was a no-op** (BUG-Y) — The `CrossEncoderReranker` was
initialized on startup, logged as loaded, and documented in the architecture overview,
but never actually called during queries or streaming. Both `query()` and
`stream_query()` passed the raw hybrid retriever directly to
`ConversationalRetrievalChain`, bypassing the reranker entirely. Only the
standalone `retrieve()` method used it — and no API endpoint calls `retrieve()`.
Fixed by introducing `RerankedRetriever`, a `BaseRetriever` wrapper that applies
hybrid search then cross-encoder reranking as one step, and using it as the
chain's retriever in both query paths.

---

## Other fixes

**`asyncio.get_event_loop()` deprecated** (BUG-Z) — `arerank()` used the
deprecated form inside a running event loop, generating `DeprecationWarning` on
Python 3.10+ and scheduled to raise in a future version. Replaced with
`asyncio.get_running_loop()`.

**Streaming `top_k` silently ignored** (BUG-AA) — The Streamlit UI's
`_api_stream()` function sent no `top_k` field to `/query/stream`, so every
streaming query fell back to the server-side default of 5 regardless of what
the caller intended. Fixed by including `top_k` in the streaming payload,
consistent with `_api_query()`.

**Thread pool leaked on shutdown** (BUG-AB) — The `CrossEncoderReranker`'s
`ThreadPoolExecutor` was never shut down on graceful server exit, leaking OS
threads each time uvicorn reloaded. Added `CrossEncoderReranker.shutdown()` and
`RAGEngine.shutdown()`, called from the FastAPI lifespan cleanup block.

**`SESSION_TTL_MINUTES` env var had no effect** (BUG-AC) — `_evict_stale()`
read the module-level constant `SESSION_TTL_MINUTES = 60` instead of
`settings.SESSION_TTL_MINUTES`, so setting the env var in `.env` was silently
ignored. Fixed to read from settings at call time.

**`bm25_docs` missing from `/health` responses** (BUG-AD) — `engine.health()`
returned `bm25_docs` and the v3.2.0 changelog documented it, but
`HealthResponse` lacked the field. FastAPI's response serializer silently
dropped it. Added `bm25_docs: str = "0"` to the Pydantic model.

**Snippet length hardcoded** (BUG-AE) — Source snippet truncation in `query()`
was hardcoded to `[:300]`. Added `SOURCE_SNIPPET_LEN: int = 300` to `Settings`
so it can be tuned via `.env`.

---

## Upgrade from v3.2.0

```bash
git pull && git checkout v3.3
# No new dependencies — requirements.txt unchanged
pip install -r requirements.txt
# or
docker compose pull && docker compose up --build
```

No database migrations required. Existing ChromaDB and BM25 pickle files are
fully compatible.

---

## Full changelog

See [CHANGELOG.md](CHANGELOG.md).
