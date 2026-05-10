# ◈ AuraRAG — Advanced Unified Retrieval Architecture

**v3.0.0** · LLM-Agnostic Enterprise RAG Pipeline  
Author: Akmal Raxmatov · [GitHub: thed700](https://github.com/thed700)

---

## What is AuraRAG?

AuraRAG is a production-grade Retrieval-Augmented Generation pipeline with:

- **Hybrid Search** — Dense vector (ChromaDB + `all-mpnet-base-v2`) + sparse BM25, fused via EnsembleRetriever
- **Cross-Encoder Re-ranking** — `ms-marco-MiniLM-L-6-v2` scores every candidate for precision
- **LLM-agnostic** — OpenAI, Anthropic Claude, Google Gemini, or local Ollama via a single provider registry
- **FastAPI backend** — async routes, SecretStr key handling, explicit CORS
- **Streamlit UI** — dark-theme chat interface with source citation chips
- **Docker-first** — multi-stage build, non-root user, named volumes for persistence

---

## Quickstart

### Local (no Docker)

```bash
# 1. Clone and create env
git clone https://github.com/thed700/enterprise-rag-pipeline.git
cd aurarag
python -m venv .venv && source .venv/bin/activate

# 2. Install CPU-only torch first (saves ~1.5 GB vs CUDA build)
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

# 3. Install the rest
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env — add API keys if you want a default; users can supply them live in the UI.

# 5. Start the backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# 6. Start the UI (new terminal)
streamlit run app/ui.py
```

Open **http://localhost:8501**

### Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

- Backend: http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

---

## Architecture

```
Upload (PDF/TXT)
      │
      ▼
  PyPDFLoader / TextLoader
      │
      ▼
  RecursiveCharacterTextSplitter  (512 tokens, 64 overlap)
      │
      ├──► ChromaDB (all-mpnet-base-v2 dense embeddings)
      └──► BM25Retriever (sparse, persisted to bm25.pkl)
                │
                ▼
          EnsembleRetriever (60% dense / 40% BM25)
                │
                ▼
      CrossEncoderReranker (ms-marco-MiniLM-L-6-v2)
                │
                ▼
    ConversationalRetrievalChain
     + ConversationBufferWindowMemory (k=20)
                │
                ▼
      OpenAI / Anthropic / Gemini / Ollama
                │
                ▼
          Answer + Source Citations
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/health` | Engine health + version |
| POST | `/ingest` | Upload PDF/TXT files for indexing |
| POST | `/query`  | Ask a question (returns answer + sources) |
| DELETE | `/memory` | Clear conversation history |

Full interactive docs at `/docs` (Swagger) or `/redoc`.

---

## Configuration

All settings are loaded from environment variables / `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `CHROMA_COLLECTION` | `aurarag` | Collection name |
| `ALLOWED_ORIGINS` | `http://localhost:8501,...` | CORS allowed origins (comma-separated) |
| `MAX_UPLOAD_MB` | `50` | Max file size for `/ingest` |
| `LOG_LEVEL` | `INFO` | Uvicorn log level |
| `OPENAI_API_KEY` | _(blank)_ | Optional default; users supply live in UI |
| `ANTHROPIC_API_KEY` | _(blank)_ | Optional default |
| `GOOGLE_API_KEY` | _(blank)_ | Optional default |

---

## Supported Providers & Models

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo |
| Anthropic | claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5, claude-3-5-sonnet, claude-3-opus |
| Google Gemini | gemini-2.5-pro, gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash |
| Local (Ollama) | llama3, mistral, mixtral, phi3, gemma2, and any locally pulled model |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## v3 Roadmap

- **v3.1** — Redis-backed per-session conversation memory (multi-worker safe)
- **v3.2** — True SSE streaming (FastAPI `StreamingResponse` + Streamlit `write_stream`)
- **v3.3** — LangGraph agentic pipeline (query rewrite → retrieve → grade → generate → reflect)
- **v3.4** — Graph-RAG (entity/relationship extraction + knowledge graph retrieval)
- **v3.5** — Vision-RAG (multimodal PDF ingestion with image captioning)
- **v3.6** — Multi-tenancy (per-tenant ChromaDB namespaces + JWT auth)

---

## Bug Fixes (v2 → v3)

| ID | Severity | Fix |
|----|----------|-----|
| BUG-01 | Critical | Unbounded conversation memory → `ConversationBufferWindowMemory(k=20)` |
| BUG-02 | Critical | Sync LLM call blocking event loop → `asyncio.to_thread()` |
| BUG-03 | High | BM25 corpus overwritten on re-ingest → `_all_docs.extend()` + disk persistence |
| BUG-04 | High | ChromaDB collection clobbered on re-ingest → incremental `add_documents()` |
| BUG-05 | High | Temp file cleanup NameError → `None`-guarded finally block |
| BUG-06 | High | API key logged in plaintext → `SecretStr` + `.get_secret_value()` |
| BUG-07 | Medium | Cross-encoder called twice per query → single rerank in retrieve path |
| BUG-08 | Medium | Multi-worker state divergence → documented; `--workers 1` enforced |
| BUG-09 | Medium | CORS wildcard + credentials (spec violation) → explicit `ALLOWED_ORIGINS` |
| BUG-10 | Low | Fake word-replay streaming → direct answer display; real SSE in v3.1 |

---

## License

MIT
