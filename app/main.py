"""
main.py — AuraRAG FastAPI Backend v3.4
Author: Akmal Raxmatov (github: thed700)

Changes v3.4:
  - Engine stored on app.state.engine so the query router's FastAPI Depends()
    can access it without a module-level global.
  - POST /query and POST /query/stream served from app/routers/query.py via
    app.include_router().  All other routes (health, providers, ingest, memory)
    remain inline for minimal change surface.
  - Lifespan teardown: engine.shutdown() still called (BUG-AB fix preserved).

Retained from v3.3 / v3.2.0 / v3.1.0:
  BUG-AB: engine.shutdown() in lifespan cleanup block.
  BUG-Q:  setup_logging() reads LOG_LEVEL from Settings.
  BUG-W:  TextLoader uses UTF-8 + autodetect_encoding.
  BUG-F:  MAX_UPLOAD_MB enforced during upload streaming.
  BUG-G:  256 KB chunked streaming write to temp file.
  BUG-H:  slowapi rate limiting on /ingest.
  BUG-O:  GET /providers returns PROVIDER_MODELS.
  BUG-P:  session_id threaded through all query endpoints.
"""

import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from app.constants import PROVIDER_MODELS
from app.engine import RAGEngine
from app.models import (
    HealthResponse,
    IngestResponse,
    ProvidersResponse,
)
from app.routers.query import router as query_router
from app.utils import APP_VERSION, get_settings, setup_logging

# BUG-Q fix (v3.2.0): setup_logging() reads LOG_LEVEL from Settings
setup_logging()
logger   = logging.getLogger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────
# RATE LIMITER
# ─────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ─────────────────────────────────────────────
# APP LIFECYCLE
# ─────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  AuraRAG — Advanced Unified Retrieval Architecture  v%s", APP_VERSION)
    logger.info("=" * 60)

    eng = RAGEngine()
    # Store on app.state so FastAPI Depends() in routers can access without
    # importing a module-level global.
    app.state.engine = eng

    yield

    logger.info("AuraRAG API shutting down.")
    # BUG-AB fix (v3.3): release CrossEncoderReranker's ThreadPoolExecutor
    # on graceful shutdown to avoid leaking OS threads.
    app.state.engine.shutdown()


app = FastAPI(
    title="AuraRAG API",
    description=(
        "Advanced Unified Retrieval Architecture v3.4. "
        "LangGraph Agentic Pipeline: Query Rewrite → Hybrid Retrieve → "
        "Document Grade → Generate → Self-Correct. "
        "LLM-agnostic: OpenAI · Anthropic · Gemini · Ollama. "
        "True SSE streaming. Per-session memory."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────

_allowed_origins = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

# ─────────────────────────────────────────────
# INCLUDE ROUTERS
# ─────────────────────────────────────────────

app.include_router(query_router)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

_CHUNK_SIZE = 1024 * 256  # 256 KB streaming write chunks  (BUG-G fix)


async def _stream_upload_to_tmp(upload: UploadFile, suffix: str) -> str:
    """
    Stream-write an uploaded file to a named temp file in 256 KB chunks.

    BUG-F / BUG-G fix (v3.1.0):
      BUG-F — MAX_UPLOAD_MB is enforced mid-stream; excess is rejected before
              the full file is read into memory.
      BUG-G — upload.read() is replaced with a chunked loop so large files
              do not spike memory.

    Returns the temp file path (caller is responsible for cleanup).
    """
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    total = 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        path = tmp.name
        while True:
            chunk = await upload.read(_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes and total > max_bytes:
                tmp.close()
                os.unlink(path)
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File '{upload.filename}' exceeds the "
                        f"{settings.MAX_UPLOAD_MB} MB upload limit."
                    ),
                )
            tmp.write(chunk)

    return path


def _engine_or_503(request: Request) -> RAGEngine:
    eng: RAGEngine | None = getattr(request.app.state, "engine", None)
    if eng is None:
        raise HTTPException(status_code=503, detail="Engine not initialised.")
    return eng


# ─────────────────────────────────────────────
# ROUTES — System
# ─────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request) -> HealthResponse:
    eng = _engine_or_503(request)
    return HealthResponse(status="ok", **eng.health())


@app.get("/providers", response_model=ProvidersResponse, tags=["System"])
async def list_providers() -> ProvidersResponse:
    """Returns the provider → model map (BUG-O fix v3.1.0)."""
    return ProvidersResponse(providers=PROVIDER_MODELS)


# ─────────────────────────────────────────────
# ROUTES — Ingestion
# ─────────────────────────────────────────────


@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestion"],
)
@limiter.limit(settings.RATE_LIMIT_INGEST)
async def ingest_documents(
    request: Request,
    files:   List[UploadFile] = File(...),
) -> IngestResponse:
    """
    Upload and index PDF or TXT documents.

    Files are streamed to disk in 256 KB chunks (BUG-G) with a configurable
    size cap (BUG-F).  TextLoader uses UTF-8 with autodetect fallback to
    handle non-UTF-8 files in C-locale containers (BUG-W).
    """
    eng = _engine_or_503(request)

    all_docs: List[Document] = []
    for upload in files:
        suffix = os.path.splitext(upload.filename or "")[-1].lower()
        if suffix not in (".pdf", ".txt"):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported type '{suffix}'. Use .pdf or .txt.",
            )

        tmp_path: str | None = None
        try:
            tmp_path = await _stream_upload_to_tmp(upload, suffix)
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
            else:
                # BUG-W fix (v3.2.0): explicit UTF-8 + autodetect fallback
                # avoids UnicodeDecodeError on non-UTF-8 text files.
                loader = TextLoader(tmp_path, encoding="utf-8", autodetect_encoding=True)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = upload.filename
            all_docs.extend(docs)
            logger.info("Loaded %d page(s) from '%s'.", len(docs), upload.filename)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    if not all_docs:
        raise HTTPException(status_code=400, detail="No content could be extracted.")

    # ingest_documents() is CPU-bound (chunk, embed, BM25 rebuild) — run in
    # a worker thread to avoid blocking the event loop.
    result = await asyncio.to_thread(eng.ingest_documents, all_docs)
    return IngestResponse(
        chunks_ingested=result["chunks_ingested"],
        duplicates_skipped=result.get("duplicates_skipped", 0),
        status=result["status"],
        message=f"Indexed {len(files)} file(s).",
    )


# ─────────────────────────────────────────────
# ROUTES — Session Memory
# ─────────────────────────────────────────────


@app.delete(
    "/memory/{session_id}",
    tags=["Session"],
    status_code=status.HTTP_204_NO_CONTENT,
)
async def clear_session_memory(session_id: str, request: Request) -> None:
    """Clear conversation history for a specific session."""
    eng = _engine_or_503(request)
    eng.clear_memory(session_id)


@app.delete(
    "/memory",
    tags=["Session"],
    status_code=status.HTTP_204_NO_CONTENT,
)
async def clear_all_memory(request: Request) -> None:
    """Clear ALL session memories (admin operation)."""
    eng = _engine_or_503(request)
    eng.clear_all_memory()
    logger.info("All session memories cleared.")
