# ◈ AuraRAG — Advanced Unified Retrieval Architecture

**v3.1.0** · LLM-Agnostic Enterprise RAG Pipeline  
Author: Akmal Raxmatov · [GitHub: thed700](https://github.com/thed700)

---

## What's New in v3.1.0

| # | Fix | Impact |
|---|-----|--------|
| BUG-A/J | `langchain-classic` is not a real PyPI package — removed | **App would not start** |
| BUG-B | `EnsembleRetriever` imported from non-existent module | **App would not start** |
| BUG-K | `ConversationBufferWindowMemory` wrong import path | **App would not start** |
| BUG-C | `_BM25_PICKLE_PATH` evaluated at import time before `.env` parsed | Wrong path in some setups |
| BUG-D | Dual `vector_store` / `_chroma_store_ref` reference — fragile alias | Silent data inconsistency |
| BUG-E | No dedup on ingest — same file uploaded twice doubled chunks | Poisoned retrieval scores |
| BUG-F | `MAX_UPLOAD_MB` defined but never enforced | Unlimited file size accepted |
| BUG-G | `upload.read()` slurped entire file into RAM | Double memory usage for large PDFs |
| BUG-H | No rate limiting on `/query` or `/ingest` | Open to DoS / quota abuse |
| BUG-I | `health()` reported `docs_indexed` from in-memory list (empty after restart) | Misleading monitoring |
| BUG-L | `HealthResponse.version` hardcoded `"3.0.0"` literal | Out of sync after bumps |
| BUG-M | Deprecated `version:` key in docker-compose | Compose warning on every up |
| BUG-N | Single Dockerfile `HEALTHCHECK` on port 8000 — UI container always fails | Container marked unhealthy |
| BUG-O | UI imported from `app.engine` — forced heavy ML deps into Streamlit container | Unnecessary bloat |
| BUG-P | Single shared memory — two browser tabs corrupted each other's history | **Critical UX bug** |
| BUG-10 | Fake word-replay "streaming" — entire response already computed | False UX promise |

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
  PyPDFLoader / TextLoader
      │
  RecursiveCharacterTextSplitter  (configurable chunk_size / overlap)
      │
  SHA-256 deduplication ── skip chunks already seen
      │
      ├──► ChromaDB (all-mpnet-base-v2, persistent, incremental add)
      └──► BM25Retriever (rank-bm25, rebuilt from cumulative corpus,
                          persisted to bm25.pkl alongside ChromaDB)
                │
                ▼
         EnsembleRetriever  (60 % dense · 40 % BM25)
                │
                ▼
      CrossEncoderReranker  (ms-marco-MiniLM-L-6-v2, ThreadPoolExecutor)
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
   to_thread)    AsyncIteratorCallbackHandler)
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

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Engine health + version + active sessions |
| GET | `/providers` | Provider → model registry (JSON) |
| POST | `/ingest` | Upload and index PDF/TXT files |
| POST | `/query` | Synchronous RAG query (returns full answer) |
| POST | `/query/stream` | SSE streaming query (token-by-token) |
| DELETE | `/memory/{session_id}` | Clear one session's history |
| DELETE | `/memory` | Clear all sessions |

Full interactive docs at `/docs`.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage |
| `CHROMA_COLLECTION` | `aurarag` | Collection name |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `SESSION_TTL_MINUTES` | `60` | Idle session eviction |
| `ALLOWED_ORIGINS` | `http://localhost:8501,...` | CORS allowed origins |
| `MAX_UPLOAD_MB` | `50` | Max file size for `/ingest` |
| `RATE_LIMIT_QUERY` | `30/minute` | `/query` rate limit per IP |
| `RATE_LIMIT_INGEST` | `10/minute` | `/ingest` rate limit per IP |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## v3 Roadmap

- **v3.1** ✅ — Per-session memory, true SSE streaming, rate limiting, dedup, 16 bug fixes
- **v3.2** — Redis-backed session memory (multi-worker safe)
- **v3.3** — LangGraph agentic pipeline (rewrite → retrieve → grade → generate → reflect)
- **v3.4** — Graph-RAG (entity/relationship knowledge graph)
- **v3.5** — Vision-RAG (multimodal PDF ingestion with image captioning)
- **v3.6** — Multi-tenancy (per-tenant ChromaDB namespaces + JWT auth)

---

## License

MIT
