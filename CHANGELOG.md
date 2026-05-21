# CHANGELOG — AuraRAG v3.5

## Bug Fixes (v3.5)

| ID       | File(s)                                | Description |
|----------|----------------------------------------|-------------|
| BUG-AK   | `app/engine/pipeline.py`               | `GRADE_THRESHOLD` was defined in `Settings` and documented in docstrings as the relevance filter threshold, but was **never applied** in `_node_grade`. The grader passed all LLM-selected documents through regardless of their relevance score. Fixed: the grader prompt now requests per-document float scores (`"scores": {index: float}`); `GRADE_THRESHOLD` is applied to filter the scored list before populating `relevant_docs`. Backward-compatible with custom prompt overrides via `keep_indices`. |
| BUG-AL   | `app/models.py`                        | `PromptOverrides` fields used `str = ""` defaults. `model_dump(exclude_none=True)` in the router does **not** filter empty strings — empty UI prompt fields silently replaced the engine's default system prompts with `""`. Fixed: fields changed to `Optional[str] = None` so `exclude_none=True` correctly drops unset overrides. |
| BUG-AM   | `app/routers/query.py`                 | The SSE `/query/stream` endpoint emitted the `[DONE]` sentinel **before** the metadata frame. Standard SSE clients stop reading at `[DONE]`; the metadata payload was always silently lost. Fixed: `meta` frame is now yielded before `[DONE]`. |
| BUG-AN   | `app/engine/pipeline.py`               | `_should_reflect()` had type hint `Literal["reflect", "__end__"]` but returned the `END` sentinel object (not the literal string). In some LangGraph versions `END` may not equal `"__end__"`. Fixed: return type widened to `str`; routing map key remains `END` for semantic clarity. |
| BUG-AO   | `app/engine/pipeline.py`               | `query()` sync wrapper raised `RuntimeError` and caught it in the same `except` block, checking `"event loop" in str(exc)` — which matched the message it just raised, turning the guard into a confusing double-raise. Fixed: the guard now detects the `"no running event loop"` message from `get_running_loop()`, swallowing only that error and re-raising all others including the advisory error. |

## Model Updates (v3.5)

| Provider       | Change |
|----------------|--------|
| **Anthropic**  | Replaced non-existent model strings `claude-opus-4-5` / `claude-sonnet-4-5` with correct versioned API identifiers (`claude-opus-4-5-20251101`, `claude-sonnet-4-5-20251022`). Added `claude-haiku-4-5-20251001`. |
| **OpenAI**     | Added `gpt-4.1` and `gpt-4.1-mini`. Default model updated to `gpt-4.1-mini`. |
| **Google**     | Added `gemini-2.0-flash`. |

---

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
