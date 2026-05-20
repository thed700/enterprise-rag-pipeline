"""
main.py — AuraRAG FastAPI Backend v4.0
Author: Akmal Raxmatov (github: thed700)

Changes v4.0:
  FEAT-DS:  /ingest route updated to use app.backend.ingest.load_uploaded_file,
            which adds support for CSV, JSON, XLSX, and Parquet uploads.
  FEAT-ENV: Dual-deployment guard: if CHROMA_PERSIST_DIR is not writable
            (Hugging Face Spaces ephemeral filesystem), the engine falls back
            to an in-memory mode and logs a warning instead of crashing.
  FEAT-DF:  file_type routing and helpful 415 error messages for all supported
            extensions.

Retained from v3.4:
  BUG-AB:  engine.shutdown() in lifespan cleanup block.
  BUG-Q:   setup_logging() reads LOG_LEVEL from Settings.
  BUG-F/G: MAX_UPLOAD_MB enforced; 256 KB chunked streaming write.
  BUG-H:   slowapi rate limiting on /ingest.
  BUG-O:   GET /providers returns PROVIDER_MODELS.
  BUG-P:   session_id threaded through all query endpoints.
  All v3.4 LangGraph pipeline and streaming fixes preserved.
"""

from __future__ import annotations

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

from langchain_core.documents import Document

from app.backend.ingest import SUPPORTED_EXTENSIONS, load_uploaded_file
from app.constants import PROVIDER_MODELS
from app.engine import RAGEngine
from app.models import (
    HealthResponse,
    IngestResponse,
    ProvidersResponse,
)
from app.routers.query import router as query_router
from app.utils import APP_VERSION, get_settings, setup_logging

setup_logging()
logger   = logging.getLogger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────
# DUAL-DEPLOYMENT DETECTION
# ─────────────────────────────────────────────

_IS_HF_SPACE = bool(os.environ.get("HF_SPACE_ID"))

if _IS_HF_SPACE:
    logger.info("Running on Hugging Face Spaces (HF_SPACE_ID detected).")
    # HF Spaces may not have write access outside /tmp; override persist dir.
    hf_persist = os.environ.get("CHROMA_PERSIST_DIR", "/tmp/aurarag_chroma")
    os.environ.setdefault("CHROMA_PERSIST_DIR", hf_persist)
    logger.info("Chroma persist dir (HF): %s", hf_persist)
else:
    logger.info("Running in local Docker mode.")

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
    logger.info("  Mode: %s", "Hugging Face Spaces" if _IS_HF_SPACE else "Local Docker")
    logger.info("=" * 60)

    eng = RAGEngine()
    app.state.engine = eng

    yield

    logger.info("AuraRAG API shutting down.")
    app.state.engine.shutdown()


app = FastAPI(
    title="AuraRAG API",
    description=(
        "Advanced Unified Retrieval Architecture v4.0. "
        "LangGraph Agentic Pipeline: Rewrite → Hybrid Retrieve → Grade → Generate → Reflect. "
        "Ingestion: PDF · TXT · CSV · JSON · XLSX · Parquet. "
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

if "*" in _allowed_origins or not _allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )
else:
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

_CHUNK_SIZE = 1024 * 256  # 256 KB streaming write chunks


async def _stream_upload_to_tmp(upload: UploadFile, suffix: str) -> str:
    """
    Stream-write an uploaded file to a named temp file in 256 KB chunks.
    Enforces MAX_UPLOAD_MB limit mid-stream (BUG-F/G fix, carried).
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
    """Returns the provider → model map."""
    return ProvidersResponse(providers=PROVIDER_MODELS)


# ─────────────────────────────────────────────
# ROUTES — Ingestion  (v4.0: multi-format)
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
    Upload and index documents.

    Supported formats: PDF, TXT, CSV, JSON, XLSX, Parquet.

    Structured data files (CSV / JSON / XLSX / Parquet) are serialised to
    natural-language row strings before embedding so they are fully searchable
    by semantic and keyword retrieval alike.

    Files are streamed to disk in 256 KB chunks with a configurable size cap.
    """
    eng = _engine_or_503(request)

    all_docs: List[Document] = []
    for upload in files:
        filename = upload.filename or "unknown"
        suffix   = os.path.splitext(filename)[-1].lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type '{suffix}' for '{filename}'. "
                    f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
                ),
            )

        tmp_path: str | None = None
        try:
            tmp_path = await _stream_upload_to_tmp(upload, suffix)
            docs     = load_uploaded_file(filename, tmp_path, suffix)
            for doc in docs:
                # Ensure source is always the original filename (loaders may
                # set it to the tmp path for PDF/TXT).
                doc.metadata.setdefault("source", filename)
            all_docs.extend(docs)
            logger.info("Loaded %d document(s) from '%s' (%s).", len(docs), filename, suffix)

        except ImportError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Missing dependency for '{suffix}' files: {exc}",
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Could not parse '{filename}': {exc}",
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    if not all_docs:
        raise HTTPException(status_code=400, detail="No content could be extracted from the uploaded files.")

    # CPU-bound ingestion in a worker thread (chunk, embed, BM25 rebuild)
    result = await asyncio.to_thread(eng.ingest_documents, all_docs)

    return IngestResponse(
        chunks_ingested    = result["chunks_ingested"],
        duplicates_skipped = result.get("duplicates_skipped", 0),
        status             = result["status"],
        message            = f"Indexed {len(files)} file(s) ({len(all_docs)} raw document(s)).",
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
