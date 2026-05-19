# ◈ AuraRAG — Advanced Unified Retrieval Architecture

**v3.4** · Agentic Enterprise RAG Pipeline (LangGraph)

Author: Akmal Raxmatov · [GitHub: thed700](https://github.com/thed700)

---

## What's New in v3.4

| ID | Change / Fix | Severity / Type |
| --- | --- | --- |
| **LANGGRAPH-CORE** | **Agentic Pipeline Transition**: Completely replaced the rigid `ConversationalRetrievalChain` with a flexible, stateful LangGraph workflow encompassing 5 multi-step agentic nodes: **Rewrite ➔ Retrieve ➔ Grade ➔ Generate ➔ Reflect**. | 🚀 Feature |
| **FEAT-OBS** | **Advanced Production Observability**: Upgraded `QueryResponse` and Server-Sent Events (SSE) payloads to deliver a detailed `pipeline_trace`, accurate tracking of `graded_chunks`, and total `reflect_loops`. Streaming sessions now append a structured JSON meta-event containing token tallies, provider metrics, and configuration footprints right after the `[DONE]` signal. | 🚀 Feature |
| **BUG-AF** | Native asynchronous execution path added to `RerankedRetriever`. It now preferentially targets `ainvoke()` with an optimized fallback to `asyncio.to_thread()` for synchronous base retrievers, completely resolving async event loop context blocks. | 🔴 High |
| **BUG-GRADER** | Implemented a fail-open execution guard within the Document Grader node. If an LLM parsing error occurs or if structural evaluation drops all assets, the pipeline smoothly cascades to a heuristic fallback rather than returning empty context or crashing. | 🔴 High |
| **BUG-STREAM** | Overhauled `stream_query()` to listen across both `messages` and `updates` streaming modes simultaneously inside LangGraph. This ensures that final pipeline state deltas are accurately tracked and saved to the session memory store during a live stream. | 🟡 Medium |
| **BUG-REFLECT** | Resolved a state-mutation bug inside the Query Reflection loop where iterative adjustment would fail to pass rewritten query state back to the grading phase properly. State dependencies now cleanly reset and populate `retrieved_docs` for subsequent passes. | 🟡 Medium |

See [CHANGELOG.md](CHANGELOG.md) for full history.

---

## Architecture

```
Upload (PDF / TXT)
      │
      ▼
  File size guard (MAX_UPLOAD_MB) ── 413 if exceeded
      │
  Chunked stream-write to temp file (256 KB chunks)
      │
  PyPDFLoader / TextLoader (UTF-8 + autodetect)
      │
  RecursiveCharacterTextSplitter (configurable chunk_size / overlap)
      │
  SHA-256 deduplication ── skip chunks already seen
  (hashes persisted to bm25.pkl — survives restarts)
      │
      ├──► ChromaDB (all-mpnet-base-v2, persistent, incremental add)
      └──► BM25Retriever (rank-bm25, rebuilt from cumulative corpus)
                │
                ▼
         EnsembleRetriever (60 % dense · 40 % BM25)
                │
                ▼
         RerankedRetriever (Handles unified sync/async rerank execution)
                │
                ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │                     LANGGRAPH AGENTIC WORKFLOW                         │
   │                                                                        │
   │                        ┌─── [Start] ───┐                               │
   │                        │               │                               │
   │                        ▼               ▼                               │
   │                [Node: Rewrite] ──► [Node: Retrieve]                    │
   │                                             │                          │
   │                                             ▼                          │
   │                                      [Node: Grade]                     │
   │                                             │                          │
   │                                             ▼                          │
   │                                     [Node: Generate]                   │
   │                                             │                          │
   │                                             ▼                          │
   │                                   ⚖️ [Should Reflect?]                 │
   │                                      /             \                   │
   │                          Yes (Loop Limit Not Met)   No / Max Reached   │
   │                                    /                 \                 │
   │                                   ▼                   ▼                │
   │                           [Node: Reflect]         [End Loop]           │
   │                                                                        │
   └────────────────────────────────────────────────────────────────────────┘
                                        │
         ┌──────────────────────────────┴──────────────────────────────┐
         │                                                             │
    /query                                                      /query/stream
    (Synchronous execution via                                  (Server-Sent Events via
     asynchronous state capture)                                 astream_events + final metrics)
         │                                                             │
         └──────────────────────────────┬──────────────────────────────┘
                                        ▼
                      OpenAI / Anthropic / Gemini / Ollama

```

---

## Quickstart

### Local

```bash
git clone https://github.com/thed700/aurarag.git
cd aurarag
python -m venv .venv && source .venv/bin/activate

# CPU-only torch (saves ~1.5 GB vs CUDA build)
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

cp .env.example .env
# Add provider API keys to your .env file

# Terminal 1: backend api engine
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# Terminal 2: frontend UI
streamlit run app/ui.py

```

Open **http://localhost:8501**

### Docker Compose

```bash
cp .env.example .env
docker compose up --build

```

* API Docs & Interactive Swagger: http://localhost:8000/docs
* Streamlit User Interface: http://localhost:8501

---

## API Reference

| Method | Path | Rate limit | Description |
| --- | --- | --- | --- |
| `GET` | `/health` | — | Engine health, system version, active session count, total `bm25_docs` |
| `GET` | `/providers` | — | Global provider-to-model registry mapping |
| `POST` | `/ingest` | 10/min | Multipart file upload and indexing engine (PDF/TXT) |
| `POST` | `/query` | 30/min | Synchronous RAG execution (Returns full agent trace and telemetry) |
| `POST` | `/query/stream` | 30/min | Real-time token stream (SSE) concluding with structural trace metadata |
| `DELETE` | `/memory/{session_id}` | — | Instantly flushes historical memory for a specific single session |
| `DELETE` | `/memory` | — | Complete system-wide clearing of all active session stores (Admin) |

### Query request fields

```jsonc
{
  "question":   "What does the policy say about overtime?",
  "top_k":      5,               // 1–20, defines bounds for retrievers & rerankers
  "provider":   "Anthropic",
  "model":      "claude-sonnet-4-6",
  "api_key":    "sk-ant-...",
  "session_id": "user-abc-123"   // Include to leverage persistent stateful graphs
}

```

### Query response fields (v3.4 Observability)

```jsonc
{
  "answer": "According to Section 4...",
  "sources": [
    { "source": "handbook.pdf", "content": "..." }
  ],
  "pipeline_trace": ["rewrite", "retrieve", "grade", "generate"], // Order of executed nodes
  "graded_chunks": {
    "total_retrieved": 5,
    "accepted_chunks": 3
  },
  "reflect_loops": 0
}

```

---

## Configuration

All system configurations and agentic parameters are derived dynamically via environment variables or loaded through `.env`.

| Variable | Default | Description |
| --- | --- | --- |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Storage path directory for vector records |
| `CHROMA_COLLECTION` | `aurarag` | Target collection name inside ChromaDB |
| `CHUNK_SIZE` | `512` | Token character ceiling for structural chunks |
| `CHUNK_OVERLAP` | `64` | Token context overlap buffer between sequential chunks |
| `SESSION_TTL_MINUTES` | `60` | Lifespan before an inactive stateful session is flushed |
| `ALLOWED_ORIGINS` | `http://localhost:8501,...` | CORS validation whitelist configuration |
| `MAX_UPLOAD_MB` | `50` | File upload ceiling limits enforced on `/ingest` |
| `RATE_LIMIT_QUERY` | `30/minute` | Rate-limiting constraints applied per IP to queries |
| `RATE_LIMIT_INGEST` | `10/minute` | Rate-limiting constraints applied per IP to file ingestion |
| `LOG_LEVEL` | `INFO` | Level settings (`DEBUG` / `INFO` / `WARNING` / `ERROR`) |
| `SOURCE_SNIPPET_LEN` | `300` | Hard character limit for returned context payloads |
| `GRADE_MAX_TOKENS` | `128` | Max output token limit for structured evaluation extraction |
| `MAX_REFLECT_LOOPS` | `2` | Global threshold governing maximum self-correction iteration runs |

---

## Running Tests

```bash
pytest tests/ -v

```

The validation suite aggressively asserts performance against all regression definitions across components—spanning `RerankedRetriever` async contexts, structured JSON parser dropguards, LangGraph stream execution handlers, multi-turn memory transformations, and validation schemas.

---

## Roadmap

| Version | Status | Theme |
| --- | --- | --- |
| v3.0 | ✅ Shipped | Multi-provider Bring-Your-Own-Key (BYOK), Hybrid search foundation |
| v3.1 | ✅ Shipped | Stateful session isolation, native SSE, global route rate-limiting mechanics |
| v3.2 | ✅ Shipped | Parameter routing reliability, streaming stability updates, deduplication persistence |
| v3.3 | ✅ Shipped | Reranker execution alignment, OS resource leak resolutions, runtime env corrections |
| **v3.4** | ✅ **Shipped** | **Agentic Pipeline Evolution: State-driven LangGraph architectures, grading safeguards, and deep instrumentation traces** |
| v3.5 | 🗓 Planned | Graph-RAG (Entity-relation mapping using knowledge extraction pipelines) |
| v3.6 | 🗓 Planned | Vision-RAG (Multimodal token extraction, chart analysis, and image descriptions) |
| v3.7 | 🗓 Planned | Secure Multi-tenancy (Namespaced vector isolation, tenant routers, and JWT validation) |

---

## License

Distributed under the MIT License. See `LICENSE` for more information.