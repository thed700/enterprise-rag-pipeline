---
title: AuraRAG — Enterprise RAG Pipeline
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ◈ AuraRAG

**Advanced Unified Retrieval Architecture · v3.5**  
Author: Akmal Raxmatov · [github.com/thed700](https://github.com/thed700)

AuraRAG is a production-grade, agentic Retrieval-Augmented Generation platform. It ingests structured and unstructured documents, indexes them in a hybrid vector + keyword store, and answers queries through a self-correcting LangGraph pipeline that rewrites, retrieves, grades, generates, and optionally reflects — all over a streaming SSE API with a dark-mode Streamlit UI.

---

## How It Works

Documents flow through a five-node LangGraph `StateGraph`. Each node is async-first and writes to shared state that is threaded end-to-end.

```
User question
     │
     ▼
[Rewrite] ── optimises the raw question into a keyword-dense search query
     │         using session history and (on retry loops) prior feedback
     ▼
[Retrieve] ── hybrid search: 60% ChromaDB MMR dense + 40% BM25 sparse
     │          → cross-encoder re-ranking → top-k candidates
     ▼
[Grade] ──── LLM scores each chunk 0.0–1.0 against the query;
     │        chunks below GRADE_THRESHOLD are dropped
     │        fallback: keyword-overlap heuristic if LLM fails
     ▼
[Generate] ── grounded answer synthesis over graded context + history
     │          appends a hidden hallucination_risk self-score (0.0–1.0)
     ▼
[Reflect?] ── if risk > 0.7 and loop budget remains:
     │          generate a refined query → re-retrieve → re-grade → re-generate
     └──────── otherwise → END
```

The graph is compiled **once at startup** and reused across all requests. Streaming uses LangGraph's native `astream()` — no bridge threads, no async queue races.

---

## Ingestion Pipeline

```
Upload (PDF / TXT / CSV / JSON / XLSX / Parquet)
        │
        ▼
  File size guard (MAX_UPLOAD_MB)
        │
  Chunked stream-write to temp file (256 KB chunks)
        │
  Loader (PyPDF / TextLoader / pandas)
        │
  RecursiveCharacterTextSplitter (CHUNK_SIZE / CHUNK_OVERLAP)
        │
  SHA-256 deduplication ── duplicate chunks skipped,
  hashes persisted to bm25.pkl so restarts don't reprocess
        │
        ├──► ChromaDB (all-mpnet-base-v2, persistent, incremental)
        └──► BM25Retriever (rank-bm25, rebuilt from cumulative corpus)
```

---

## Quickstart

### Local

```bash
git clone https://github.com/thed700/aurarag.git
cd aurarag

python -m venv .venv && source .venv/bin/activate

# Install CPU-only PyTorch first to avoid pulling the multi-GB CUDA build
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

cp .env.example .env
# Edit .env — add your provider API key(s)

# Terminal 1 — backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# Terminal 2 — UI
streamlit run app/ui.py
```

Open **http://localhost:8501** for the UI or **http://localhost:8000/docs** for the interactive API explorer.

### Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

---

## API Reference

| Method | Path | Rate limit | Description |
|--------|------|-----------|-------------|
| `GET` | `/health` | — | Engine status, version, indexed doc count, active session count |
| `GET` | `/providers` | — | Provider → model registry |
| `POST` | `/ingest` | 10 / min | Upload and index files (PDF, TXT, CSV, JSON, XLSX, Parquet) |
| `POST` | `/query` | 30 / min | Full agentic RAG query, returns complete response with trace |
| `POST` | `/query/stream` | 30 / min | Same pipeline, token-by-token SSE stream |
| `DELETE` | `/memory/{session_id}` | — | Clear one session's conversation history |
| `DELETE` | `/memory` | — | Clear all session memory |

### Query request

```jsonc
POST /query
{
  "question":   "What does the policy say about remote work?",
  "top_k":      5,              // chunks returned to the generate node (1–20)
  "provider":   "Anthropic",
  "model":      "claude-sonnet-4-5-20251022",
  "api_key":    "sk-ant-...",
  "session_id": "user-abc-123", // omit for stateless one-shot queries
  "system_prompts": {           // optional — override any node's system prompt
    "rewrite":  "...",
    "grade":    "...",
    "generate": "...",
    "reflect":  "..."
  }
}
```

### Query response

```jsonc
{
  "answer": "According to Section 4...",
  "sources": [
    { "content": "...excerpt...", "metadata": { "source": "handbook.pdf", "page": 12 } }
  ],
  "chat_history":   ["Human: ...", "Assistant: ..."],
  "session_id":     "user-abc-123",
  "pipeline_trace": ["rewrite", "retrieve", "grade", "generate"],
  "graded_chunks":  3,
  "reflect_loops":  0
}
```

### Streaming

`POST /query/stream` returns a `text/event-stream`. Events arrive in this order:

```
data: {"token": "According"}
data: {"token": " to"}
...
data: {"meta": {"session_id": "...", "token_count": 142, "provider": "...", "model": "...", "top_k": 5}}
data: [DONE]
```

---

## Supported Providers & Models

| Provider | Models |
|----------|--------|
| **OpenAI** | `gpt-4.1`, `gpt-4.1-mini`, `gpt-4o`, `gpt-4o-mini` |
| **Anthropic** | `claude-opus-4-5-20251101`, `claude-sonnet-4-5-20251022`, `claude-3-5-sonnet-20241022`, `claude-haiku-4-5-20251001`, `claude-3-5-haiku-20241022` |
| **Google Gemini** | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-1.5-flash` |
| **Ollama (local)** | `llama3`, `mistral`, or any model string your Ollama instance serves |

Keys are supplied per-request through the UI or API — nothing is stored server-side.

---

## Configuration

All settings are read from environment variables or a `.env` file. The `Settings` object is cached after first load; call `get_settings.cache_clear()` in tests that need to vary values.

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path. Falls back to a temp directory if not writable. |
| `CHROMA_COLLECTION` | `aurarag` | ChromaDB collection name |
| `CHUNK_SIZE` | `512` | Characters per text chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `SESSION_TTL_MINUTES` | `60` | Idle session eviction timeout |
| `MAX_UPLOAD_MB` | `50` | Per-file upload size cap (0 = unlimited) |
| `ALLOWED_ORIGINS` | `http://localhost:8501,...` | CORS allowed origins (comma-separated) |
| `RATE_LIMIT_QUERY` | `30/minute` | Per-IP query rate limit |
| `RATE_LIMIT_INGEST` | `10/minute` | Per-IP ingest rate limit |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `SOURCE_SNIPPET_LEN` | `300` | Max characters per source chunk in responses |
| `GRADE_THRESHOLD` | `0.5` | Minimum relevance score (0.0–1.0) for a chunk to pass the Grade node |
| `REFLECT_ENABLED` | `true` | Enable / disable the Reflect self-correction node |
| `MAX_REFLECT_LOOPS` | `1` | Maximum reflect → retrieve → grade → generate iterations |
| `REWRITE_MAX_TOKENS` | `128` | Token budget for the Query Rewrite node |
| `GRADE_MAX_TOKENS` | `64` | Token budget for each Document Grade call |

---

## Project Structure

```
aurarag/
├── app/
│   ├── main.py               # FastAPI app, lifespan, /ingest, /health, /providers
│   ├── models.py             # Pydantic request/response schemas
│   ├── utils.py              # Settings (pydantic-settings), logging setup
│   ├── constants.py          # Re-exports from app.backend.models (compat shim)
│   ├── ui.py                 # Streamlit dark-mode chat UI
│   ├── engine/
│   │   ├── __init__.py       # Public re-exports for backward compat
│   │   └── pipeline.py       # LangGraph graph, all nodes, RAGEngine class
│   ├── backend/
│   │   ├── models.py         # Provider/model registry, key validation
│   │   └── ingest.py         # File loaders (PDF, TXT, CSV, JSON, XLSX, Parquet)
│   └── routers/
│       └── query.py          # /query and /query/stream endpoints
├── tests/
│   └── test_engine.py
├── data/                     # ChromaDB persistence + BM25 pickle
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers `RerankedRetriever` async paths, `_safe_json_object` parser edge cases, LangGraph stream event handling, multi-turn session memory, and Pydantic schema validation.

---

## Roadmap

| Version | Status | Focus |
|---------|--------|-------|
| v3.0 | ✅ Shipped | Multi-provider BYOK, hybrid search |
| v3.1 | ✅ Shipped | Stateful sessions, SSE streaming, rate limiting |
| v3.2 | ✅ Shipped | Parameter routing, streaming stability, deduplication persistence |
| v3.3 | ✅ Shipped | Reranker wiring, resource leak fixes, runtime config corrections |
| v3.4 | ✅ Shipped | LangGraph agentic pipeline, document grading, observability traces |
| **v3.5** | ✅ **Shipped** | **Bug fixes: GRADE_THRESHOLD enforcement, PromptOverrides empty-string override, SSE meta ordering, sync wrapper guard, correct model API strings** |
| v3.6 | 🗓 Planned | Graph-RAG — entity/relation extraction, knowledge graph traversal |
| v3.7 | 🗓 Planned | Vision-RAG — multimodal ingestion, chart and image understanding |
| v3.8 | 🗓 Planned | Multi-tenancy — namespaced vector isolation, JWT auth, tenant routing |

---

## License

MIT — see [LICENSE](LICENSE).