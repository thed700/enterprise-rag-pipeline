---
title: AuraRAG — Enterprise RAG Pipeline
emoji: ◈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

<div align="center">

# ◈ AuraRAG

**Advanced Unified Retrieval Architecture**

`v3.6` · Production-grade · Multi-provider · Agentic · Streaming

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776ab.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-6e9fff.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-ff4b4b.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com)

*Author: [Akmal Raxmatov](https://github.com/thed700)*

</div>

---

AuraRAG is a production-grade, agentic Retrieval-Augmented Generation platform built for enterprise document intelligence. It ingests structured and unstructured documents, indexes them in a hybrid dense + sparse retrieval store, and answers queries through a self-correcting five-node LangGraph pipeline — all over a true SSE streaming API with a dark-mode Streamlit UI.

Bring your own keys. Nothing is stored server-side.

---

## How It Works

Documents flow through a compiled `StateGraph`. Each node is async-first and writes to a shared typed state that threads end-to-end through the pipeline.

```
User question
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  [Rewrite]   Reformulate the raw question into a keyword-dense  │
│              search query using session history + prior feedback │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  [Retrieve]  Hybrid search: 60% ChromaDB MMR dense +            │
│              40% BM25 sparse → cross-encoder re-ranking         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  [Grade]     LLM scores each chunk 0.0–1.0 against the query.   │
│              Chunks below GRADE_THRESHOLD are filtered out.     │
│              Keyword-overlap heuristic fallback if LLM fails.   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  [Generate]  Grounded answer synthesis over graded context      │
│              and session history. Appends a hidden              │
│              hallucination_risk self-score (0.0 – 1.0).        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             │  risk > 0.7              │  risk ≤ 0.7
             ▼  and loops remain        ▼
┌─────────────────────┐           ┌─────┐
│  [Reflect]          │           │ END │
│  Refine the query   │           └─────┘
│  → re-retrieve      │
│  → re-grade         │
│  → re-generate      │
└─────────────────────┘
```

The graph is compiled **once at startup** and reused across all requests. Streaming uses LangGraph's native `astream()` — no bridge threads, no async queue races, no hidden JSON bleed into the chat.

---

## Ingestion Pipeline

```
Upload  PDF / TXT / CSV / JSON / XLSX / Parquet
          │
          ▼  File size guard  (MAX_UPLOAD_MB)
          │
          ▼  Chunked stream-write to temp file  (256 KB blocks)
          │
          ▼  Loader
          │   ├─ PDF      → PyPDFLoader (page-level)
          │   ├─ TXT      → TextLoader (UTF-8 + autodetect)
          │   ├─ CSV      → pandas → row-level text documents
          │   ├─ JSON     → pd.json_normalize → row-level text
          │   ├─ XLSX/XLS → per-sheet row-level text documents
          │   └─ Parquet  → pandas → row-level text documents
          │
          ▼  RecursiveCharacterTextSplitter
          │   (CHUNK_SIZE chars, CHUNK_OVERLAP overlap)
          │
          ▼  SHA-256 deduplication
          │   duplicate chunks skipped + hashes persisted to bm25.pkl
          │   so restarts never reprocess the same content
          │
          ├──► ChromaDB  (all-mpnet-base-v2, persistent, incremental)
          └──► BM25Retriever  (rank-bm25, rebuilt from cumulative corpus)
```

---

## Quickstart

### Local development

```bash
git clone https://github.com/thed700/enterprise-rag-pipeline.git
cd enterprise-rag-pipeline

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install CPU-only PyTorch first to avoid the multi-GB CUDA build
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

cp .env.example .env
# Open .env and set at least one provider key

# Terminal 1 — backend API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# Terminal 2 — Streamlit UI
streamlit run app/ui.py
```

Open **http://localhost:8501** for the chat UI or **http://localhost:8000/docs** for the interactive API explorer.

### Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

The `api` service runs on port `8000`, the `ui` service on port `8501`. The `ui` container waits for the API health check before starting.

### Hugging Face Spaces

The repo ships with a `Dockerfile` and `entrypoint.sh` configured for HF Spaces. Push to a Space set to **Docker** SDK — the entrypoint starts FastAPI on port `8000`, waits for it to become healthy, then starts Streamlit on port `7860` (the HF default routing port).

---

## API Reference

| Method | Path | Rate limit | Description |
|--------|------|-----------|-------------|
| `GET` | `/health` | — | Engine status, version, indexed doc count, active sessions |
| `GET` | `/providers` | — | Full provider → model registry |
| `POST` | `/ingest` | 10 / min | Upload and index files |
| `POST` | `/query` | 30 / min | Full agentic RAG query, complete response with pipeline trace |
| `POST` | `/query/stream` | 30 / min | Same pipeline, token-by-token SSE stream with full meta frame |
| `DELETE` | `/memory/{session_id}` | — | Clear one session's conversation history |
| `DELETE` | `/memory` | — | Clear all session memory |

### Request body — `/query` and `/query/stream`

```jsonc
{
  "question":   "What does the policy say about remote work?",
  "top_k":      5,              // chunks passed to Generate node (1 – 20)
  "provider":   "Anthropic",
  "model":      "claude-sonnet-4-6",
  "api_key":    "sk-ant-...",   // never stored server-side
  "session_id": "user-abc-123", // omit for stateless one-shot queries
  "system_prompts": {           // optional — override any node's system prompt
    "rewrite":  "...",
    "grade":    "...",
    "generate": "...",
    "reflect":  "..."
  }
}
```

### Response body — `/query`

```jsonc
{
  "answer": "According to Section 4 of the Employee Handbook...",
  "sources": [
    {
      "content": "...relevant excerpt...",
      "metadata": { "source": "handbook.pdf", "page": 12 }
    }
  ],
  "chat_history":   ["Human: ...", "Assistant: ..."],
  "session_id":     "user-abc-123",
  "pipeline_trace": ["rewrite", "retrieve", "grade", "generate"],
  "graded_chunks":  3,
  "reflect_loops":  0
}
```

### SSE event sequence — `/query/stream`

```
data: {"token": "According"}
data: {"token": " to"}
data: {"token": " Section"}
...
data: {"meta": {
  "session_id": "user-abc-123",
  "token_count": 142,
  "provider": "Anthropic",
  "model": "claude-sonnet-4-6",
  "top_k": 5,
  "pipeline_trace": ["rewrite", "retrieve", "grade", "generate"],
  "graded_chunks": 3,
  "reflect_loops": 0,
  "sources": [{ "content": "...", "metadata": {...} }]
}}
data: [DONE]
```

The `meta` frame now carries the **full** pipeline metadata including `sources`, `pipeline_trace`, `graded_chunks`, and `reflect_loops` — source cards and trace badges render correctly in streaming mode.

---

## Supported Providers & Models

Keys are supplied per-request through the UI or API — nothing is stored server-side.

| Provider | Models |
|----------|--------|
| **OpenAI** | `gpt-4.1`, `gpt-4.1-mini`, `gpt-4o`, `gpt-4o-mini` |
| **Anthropic** | `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-sonnet-4-5`, `claude-haiku-4-5`, `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022` |
| **Google Gemini** | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-1.5-flash` |
| **Ollama (local)** | `llama3`, `mistral`, or any model string your Ollama instance serves |

---

## Configuration

All settings are read from environment variables or a `.env` file. The `Settings` object is cached after first load via `@lru_cache`; call `get_settings.cache_clear()` in tests that need to vary values.

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path. Falls back to a temp dir if the path is not writable (e.g. HF Spaces read-only FS). |
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
| `GRADE_THRESHOLD` | `0.5` | Minimum relevance score (0.0 – 1.0) for a chunk to pass the Grade node |
| `REFLECT_ENABLED` | `true` | Enable / disable the Reflect self-correction node |
| `MAX_REFLECT_LOOPS` | `1` | Maximum reflect → retrieve → grade → generate iterations |
| `REWRITE_MAX_TOKENS` | `128` | Token budget for the Query Rewrite node |
| `GRADE_MAX_TOKENS` | `64` | Token budget for each Document Grade call |

---

## Project Structure

```
enterprise-rag-pipeline/
├── app/
│   ├── main.py               # FastAPI app, lifespan, /ingest, /health, /providers
│   ├── models.py             # Pydantic request/response schemas
│   ├── utils.py              # Settings (pydantic-settings), setup_logging
│   ├── constants.py          # Re-exports from app.backend.models (compat shim)
│   ├── ui.py                 # Streamlit dark-mode chat UI
│   ├── engine/
│   │   ├── __init__.py       # Public re-exports for backward compatibility
│   │   └── pipeline.py       # LangGraph graph, all nodes, RAGEngine class
│   ├── backend/
│   │   ├── models.py         # Provider/model registry, key format validation
│   │   └── ingest.py         # File loaders (PDF, TXT, CSV, JSON, XLSX, Parquet)
│   └── routers/
│       └── query.py          # /query and /query/stream endpoints
├── tests/
│   └── test_engine.py        # Regression suite (40+ tests)
├── data/                     # ChromaDB persistence + BM25 pickle (gitignored)
├── .github/
│   └── workflows/
│       └── build-artifacts.yml  # Windows EXE + Android APK CI
├── docker-compose.yml
├── Dockerfile                # Multi-stage, CPU-only torch, HF Spaces ready
├── entrypoint.sh             # Orchestrates API + UI for Docker / HF Spaces
├── requirements.txt
├── .env.example
└── CHANGELOG.md
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

The test suite includes:

- **Import chain** — all modules load cleanly with no heavy dependency side-effects
- **Provider registry** — correct model IDs for all four providers, no stale strings
- **CrossEncoderReranker** — sync and async reranking, empty input, executor shutdown
- **RerankedRetriever** — used in both `query()` and `stream_query()` paths (BUG-Y regression)
- **SessionMemoryStore** — session isolation, TTL eviction, clear cleans `_last_access`
- **Deduplication** — second ingest of the same file skips all chunks, `_seen_hashes` persisted
- **Pydantic schemas** — `api_key` SecretStr masking, `bm25_docs` field, `message` non-null
- **Settings** — `LOG_LEVEL`, `SOURCE_SNIPPET_LEN`, `SESSION_TTL_MINUTES` all respected at runtime

---

## Roadmap

| Version | Status | Focus |
|---------|--------|-------|
| v3.0 | ✅ Shipped | Multi-provider BYOK, hybrid search, dark-mode UI |
| v3.1 | ✅ Shipped | Stateful sessions, SSE streaming, rate limiting |
| v3.2 | ✅ Shipped | Parameter routing, streaming stability, deduplication persistence |
| v3.3 | ✅ Shipped | Reranker wiring (BUG-Y), resource leak fixes, runtime config corrections |
| v3.4 | ✅ Shipped | Full LangGraph agentic pipeline: Rewrite → Retrieve → Grade → Generate → Reflect |
| v3.5 | ✅ Shipped | Grade threshold enforcement, PromptOverrides fix, SSE meta ordering, sync guard |
| **v3.6** | ✅ **Shipped** | **Avatar crash fix, meta block leak fix, SSE sources fix, correct Anthropic model IDs, removed unused 2 GB dependency, Android SDK license fix** |
| v3.7 | 🗓 Planned | Graph-RAG — entity/relation extraction, knowledge graph traversal |
| v3.8 | 🗓 Planned | Vision-RAG — multimodal ingestion, chart and image understanding |
| v3.9 | 🗓 Planned | Multi-tenancy — namespaced vector isolation, JWT auth, tenant routing |

---

## License

MIT — see [LICENSE](LICENSE).
