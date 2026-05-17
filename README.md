# ◈ AuraRAG — Advanced Unified Retrieval Architecture

**v3.3** · LLM-Agnostic Enterprise RAG Pipeline  
Author: Akmal Raxmatov · [GitHub: thed700](https://github.com/thed700)

---

## What's New in v3.3

| ID | Fix | Severity |
|----|-----|----------|
| BUG-Y | `CrossEncoderReranker` was initialized but **completely bypassed** in `query()` and `stream_query()` — both used `_build_hybrid_retriever()` directly. Reranker only ran in a standalone method no endpoint calls. Fixed with `RerankedRetriever(BaseRetriever)` now used in both paths. | 🔴 High |
| BUG-Z | `arerank()` called deprecated `asyncio.get_event_loop()` inside a running loop — `DeprecationWarning` on Python 3.10+, will raise in future. Replaced with `asyncio.get_running_loop()`. | 🟡 Medium |
| BUG-AA | `_api_stream()` sent no `top_k` to `/query/stream` — every streaming query silently fell back to server default of 5. `top_k` now included in the payload. | 🟡 Medium |
| BUG-AB | `CrossEncoderReranker._executor` (ThreadPoolExecutor) never shut down — leaked OS threads on every hot-reload. Added `shutdown()` to reranker + engine, called from FastAPI lifespan cleanup. | 🟡 Medium |
| BUG-AC | `_evict_stale()` read module-level constant `SESSION_TTL_MINUTES = 60` instead of `settings.SESSION_TTL_MINUTES` — setting the env var had zero effect. Now reads from settings at call time. | 🟡 Medium |
| BUG-AD | `HealthResponse` was missing the `bm25_docs` field that `engine.health()` returns — FastAPI silently dropped it from every `/health` response. Added `bm25_docs: str = "0"` to the model. | 🟡 Medium |
| BUG-AE | Source snippet truncation in `query()` was hardcoded to `[:300]`. Added `SOURCE_SNIPPET_LEN: int = 300` to `Settings` — tunable via `.env`. | 🟢 Low |

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
  RecursiveCharacterTextSplitter  (configurable chunk_size / overlap)
      │
  SHA-256 deduplication ── skip chunks already seen
  (hashes persisted to bm25.pkl — survives restarts)
      │
      ├──► ChromaDB (all-mpnet-base-v2, persistent, incremental add)
      └──► BM25Retriever (rank-bm25, rebuilt from cumulative corpus,
                          atomic pickle write alongside ChromaDB)
                │
                ▼
         EnsembleRetriever  (60 % dense · 40 % BM25)
                │
                ▼
      RerankedRetriever  ← NEW in v3.3 (BUG-Y fix)
      (wraps EnsembleRetriever + CrossEncoderReranker as one BaseRetriever
       so reranking is active inside ConversationalRetrievalChain, not bypassed)
                │
                ▼
   ConversationalRetrievalChain
    + SessionMemoryStore  (per-session ConversationBufferWindowMemory k=20,
                           TTL-based eviction from settings, keyed by session_id)
                │
         ┌──────┴──────┐
         │             │
    /query          /query/stream
    (sync,          (true SSE via
   to_thread)    AsyncIteratorCallbackHandler
                   + exception propagation)
         │             │
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
# Optionally add provider keys to .env

# Terminal 1: backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# Terminal 2: UI
streamlit run app/ui.py
```

Open **http://localhost:8501**

### Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

- API + Swagger: http://localhost:8000/docs
- Streamlit UI:  http://localhost:8501

---

## API Reference

| Method | Path | Rate limit | Description |
|--------|------|-----------|-------------|
| `GET` | `/health` | — | Engine health, version, session count, bm25_docs |
| `GET` | `/providers` | — | Provider → model registry |
| `POST` | `/ingest` | 10/min | Upload & index PDF/TXT (multipart) |
| `POST` | `/query` | 30/min | Synchronous RAG query |
| `POST` | `/query/stream` | 30/min | SSE streaming query (token-by-token) |
| `DELETE` | `/memory/{session_id}` | — | Clear one session's history |
| `DELETE` | `/memory` | — | Clear all sessions (admin) |

Interactive docs at **`/docs`** (Swagger UI).

### Query request fields

```jsonc
{
  "question":   "What does the policy say about overtime?",
  "top_k":      5,          // 1–20, controls reranker + source count
  "provider":   "Anthropic",
  "model":      "claude-sonnet-4-6",
  "api_key":    "sk-ant-...",
  "session_id": "user-abc-123"   // omit for stateless single-turn
}
```

---

## Configuration

All settings are read from `.env` (or environment variables).

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `CHROMA_COLLECTION` | `aurarag` | Collection name |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `SESSION_TTL_MINUTES` | `60` | Idle session eviction |
| `ALLOWED_ORIGINS` | `http://localhost:8501,...` | CORS allowed origins |
| `MAX_UPLOAD_MB` | `50` | Max file size for `/ingest` |
| `RATE_LIMIT_QUERY` | `30/minute` | `/query` rate limit per IP |
| `RATE_LIMIT_INGEST` | `10/minute` | `/ingest` rate limit per IP |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `SOURCE_SNIPPET_LEN` | `300` | Source snippet chars returned per chunk *(new in v3.3)* |

---

## Running Tests

```bash
pytest tests/ -v
```

The suite covers all 7 v3.3 regressions (BUG-Y through BUG-AE) plus retained coverage for imports, reranker, session memory, TTL eviction, deduplication, hash persistence, top_k forwarding, health reporting, and all Pydantic schemas.

---

## Roadmap

| Version | Status | Theme |
|---------|--------|-------|
| v3.0 | ✅ shipped | Multi-provider BYOK, hybrid search |
| v3.1 | ✅ shipped | Per-session memory, true SSE, rate limiting, 16 bug fixes |
| v3.2 | ✅ shipped | 8 bug fixes: top_k, streaming safety, hash persistence, encoding, session clear, log level, model IDs |
| **v3.3** | ✅ **shipped** | **7 bug fixes: reranker bypass, thread leak, TTL env var, bm25_docs field, stream top_k, get_running_loop, snippet length** |
| v3.4 | 🗓 planned | LangGraph agentic pipeline (rewrite → retrieve → grade → generate → reflect) |
| v3.5 | 🗓 planned | Graph-RAG (entity/relationship knowledge graph) |
| v3.6 | 🗓 planned | Vision-RAG (multimodal PDF ingestion with image captioning) |
| v3.7 | 🗓 planned | Multi-tenancy (per-tenant ChromaDB namespaces + JWT auth) |

---

## License

MIT
