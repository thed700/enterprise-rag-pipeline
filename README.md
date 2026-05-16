# ◈ AuraRAG — Advanced Unified Retrieval Architecture

**v3.2.0** · LLM-Agnostic Enterprise RAG Pipeline  
Author: Akmal Raxmatov · [GitHub: thed700](https://github.com/thed700)

---

## What's New in v3.2.0

| ID | Fix | Severity |
|----|-----|----------|
| BUG-S | `top_k` forwarded through full API → engine → reranker chain (was silently dropped) | 🔴 High |
| BUG-V | `stream_query()` chain exceptions now surface as SSE error frames (was silent hang) | 🔴 High |
| BUG-X | `_seen_hashes` persisted to pickle + atomic write — dedup now survives restarts & upgrades | 🔴 High |
| BUG-W | `TextLoader` uses `encoding="utf-8", autodetect_encoding=True` — no more `UnicodeDecodeError` | 🔴 High |
| BUG-U | `SessionMemoryStore.clear()` removes `_last_access` entry — cleared sessions count correctly | 🟡 Medium |
| BUG-Q | `setup_logging()` now reads `Settings.LOG_LEVEL` — `LOG_LEVEL=DEBUG` actually works | 🟡 Medium |
| BUG-R | Anthropic model IDs corrected: `claude-opus-4-6`, `claude-sonnet-4-6` (stale 4-5 aliases removed) | 🟡 Medium |
| BUG-T | `IngestResponse.message` typed `str` not `Optional[str]` — matches actual behaviour | 🟢 Low |

See [CHANGELOG.md](CHANGELOG.md) for the full diff.

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
      CrossEncoderReranker  (ms-marco-MiniLM-L-6-v2, ThreadPoolExecutor)
      top_k honoured end-to-end from API request → reranker → source slice
                │
                ▼
   ConversationalRetrievalChain
    + SessionMemoryStore  (per-session ConversationBufferWindowMemory k=20,
                           TTL-based eviction, keyed by session_id)
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
| `REDIS_URL` | _(unset)_ | Optional Redis for session persistence |

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output: **all tests pass**. The suite covers import hygiene, reranker, session memory (including TTL eviction and the BUG-U clear fix), dedup + hash persistence (BUG-X), top_k forwarding (BUG-S), health reporting, and all Pydantic schemas.

---

## Roadmap

| Version | Status | Theme |
|---------|--------|-------|
| v3.0 | ✅ shipped | Multi-provider BYOK, hybrid search |
| v3.1 | ✅ shipped | Per-session memory, true SSE, rate limiting, 16 bug fixes |
| **v3.2** | ✅ **shipped** | **8 bug fixes: top_k, streaming safety, hash persistence, encoding, session clear, log level, model IDs** |
| v3.3 | 🗓 planned | LangGraph agentic pipeline (rewrite → retrieve → grade → generate → reflect) |
| v3.4 | 🗓 planned | Graph-RAG (entity/relationship knowledge graph) |
| v3.5 | 🗓 planned | Vision-RAG (multimodal PDF ingestion with image captioning) |
| v3.6 | 🗓 planned | Multi-tenancy (per-tenant ChromaDB namespaces + JWT auth) |

---

## License

MIT
