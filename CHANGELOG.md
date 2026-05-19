# CHANGELOG — AuraRAG v3.4

## Architecture

### LangGraph Agentic Pipeline
Replaced the monolithic `ConversationalRetrievalChain` with a compiled
LangGraph `StateGraph` comprising five nodes and conditional edges:

```
START → [rewrite] → [retrieve] → [grade] → [generate] → END
                                                ↓ (conditional)
                                           [reflect] → [grade]  (loops up to MAX_REFLECT_LOOPS)
```

| Node       | Responsibility                                                    |
|------------|-------------------------------------------------------------------|
| `rewrite`  | Transform the raw user question into a search-optimised query    |
| `retrieve` | Hybrid Search (60% ChromaDB MMR + 40% BM25) + Cross-Encoder rank |
| `grade`    | Parallel LLM relevance filter; keyword heuristic fallback         |
| `generate` | Grounded answer synthesis with hallucination_risk self-score      |
| `reflect`  | Refined query generation + re-retrieval when risk > 0.7           |

### Module Layout
`app/engine/` is now a package; `pipeline.py` contains the full engine.  
`app/engine/__init__.py` re-exports the public API for backward compatibility.

---

## Bug Fixes (v3.4)

| ID       | Description                                                           |
|----------|-----------------------------------------------------------------------|
| BUG-AF   | `RerankedRetriever._aget_relevant_documents()` called sync `invoke()` inside async context, blocking the event loop. Fixed: async path uses `ainvoke()` when available, falls back to `asyncio.to_thread()`. |
| BUG-AG   | `stream_query()` rebuilt `ConversationalRetrievalChain` on every request. Fixed: LangGraph compiled once at `__init__()`. |
| BUG-AH   | SSE streaming bridged event loop and worker thread via `AsyncIteratorCallbackHandler`, creating a data race on `asyncio.Queue` at shutdown. Fixed: LangGraph's native `astream()` is fully async-first. |

---

## Bug Fixes Carried Forward from v3.3

| ID       | Description                                                            |
|----------|------------------------------------------------------------------------|
| BUG-Y    | CrossEncoderReranker completely bypassed in `query()` / `stream_query()`. Fixed: `RerankedRetriever` wraps hybrid search + reranking in one `BaseRetriever`. |
| BUG-Z    | `arerank()` used deprecated `asyncio.get_event_loop()`. Fixed: `asyncio.get_running_loop()`. |
| BUG-AB   | `CrossEncoderReranker._executor` never shut down. Fixed: `shutdown()` called from lifespan cleanup. |
| BUG-AC   | `_evict_stale()` used module-level `SESSION_TTL_MINUTES` constant. Fixed: reads `get_settings().SESSION_TTL_MINUTES`. |
| BUG-AD   | `HealthResponse` was missing `bm25_docs`. Fixed: field added. |
| BUG-AE   | Source snippet length hardcoded to `300`. Fixed: reads `settings.SOURCE_SNIPPET_LEN`. |
| BUG-S    | `top_k` dropped between API and engine. Fixed: forwarded end-to-end. |
| BUG-U    | `SessionMemoryStore.clear()` left stale `_last_access` entry. Fixed. |
| BUG-V    | Streaming chain exceptions silently swallowed. Fixed: propagated to SSE error frame. |
| BUG-W    | `TextLoader` raised `UnicodeDecodeError` on non-UTF-8 files. Fixed: `encoding="utf-8", autodetect_encoding=True`. |
| BUG-X    | `_seen_hashes` not persisted in BM25 pickle. Fixed: atomic write with hashes. |

---

## New Settings (`.env`)

| Variable             | Default | Description                                       |
|----------------------|---------|---------------------------------------------------|
| `GRADE_THRESHOLD`    | `0.5`   | Minimum relevance score for the Document Grader   |
| `REFLECT_ENABLED`    | `true`  | Enable/disable the Reflect node                   |
| `MAX_REFLECT_LOOPS`  | `1`     | Maximum self-correction loops per query           |
| `REWRITE_MAX_TOKENS` | `128`   | Token budget for the Query Rewrite LLM call       |
| `GRADE_MAX_TOKENS`   | `64`    | Token budget for each Document Grader LLM call    |

---

## API Changes

`QueryResponse` now includes three observability fields (backward-compatible
additions — existing clients simply ignore them):

| Field            | Type        | Description                               |
|------------------|-------------|-------------------------------------------|
| `pipeline_trace` | `List[str]` | Ordered node names executed               |
| `graded_chunks`  | `int`       | Chunks that passed the Document Grader    |
| `reflect_loops`  | `int`       | Self-correction loops performed           |

The SSE stream now emits a final `{"meta": {...}}` event after `[DONE]` with
`session_id`, `token_count`, `provider`, `model`, and `top_k`.

---

## Dependencies Added

```
langgraph>=0.2.0
langgraph-checkpoint>=1.0.0
```
