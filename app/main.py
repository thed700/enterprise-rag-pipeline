"""
main.py — AuraRAG FastAPI backend.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.backend.ingest import load_documents_from_file
from app.constants import PROVIDER_MODELS
from app.engine_core import RAGEngine
from app.models import HealthResponse, IngestResponse, ProvidersResponse
from app.routers.query import router as query_router
from app.utils import APP_VERSION, get_settings, setup_logging

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  AuraRAG — Advanced Unified Retrieval Architecture  v%s", APP_VERSION)
    logger.info("=" * 60)
    app.state.engine = RAGEngine()
    yield
    logger.info("AuraRAG API shutting down.")
    app.state.engine.shutdown()

app = FastAPI(
    title="AuraRAG API",
    description=(
        f"Advanced Unified Retrieval Architecture v{APP_VERSION}. "
        "LangGraph Agentic Pipeline with streaming support."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_allowed_origins = [
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    ).split(",")
    if origin.strip()
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

app.include_router(query_router)

_CHUNK_SIZE = 1024 * 256

async def _stream_upload_to_tmp(upload: UploadFile, suffix: str) -> str:
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

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request) -> HealthResponse:
    eng = _engine_or_503(request)
    return HealthResponse(status="ok", **eng.health())

@app.get("/providers", response_model=ProvidersResponse, tags=["System"])
async def list_providers() -> ProvidersResponse:
    return ProvidersResponse(providers=PROVIDER_MODELS)

@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestion"],
)
@limiter.limit(settings.RATE_LIMIT_INGEST)
async def ingest_documents(
    request: Request,
    files: List[UploadFile] = File(...),
) -> IngestResponse:
    """
    Upload and index structured documents. PDF/TXT/CSV/JSON/XLSX/Parquet are supported.
    """
    eng = _engine_or_503(request)

    upload_records: list[tuple[str, str]] = []
    all_docs = []

    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in {".pdf", ".txt", ".csv", ".json", ".xlsx", ".xls", ".parquet"}:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported type '{suffix}'. Use PDF, TXT, CSV, JSON, XLSX, or Parquet.",
            )

        tmp_path: str | None = None
        try:
            tmp_path = await _stream_upload_to_tmp(upload, suffix)
            docs = load_documents_from_file(tmp_path, original_name=upload.filename)
            all_docs.extend(docs)
            logger.info("Loaded %d document chunk(s) from '%s'.", len(docs), upload.filename)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    if not all_docs:
        raise HTTPException(status_code=400, detail="No content could be extracted.")

    result = await asyncio.to_thread(eng.ingest_documents, all_docs)
    return IngestResponse(
        chunks_ingested=result["chunks_ingested"],
        duplicates_skipped=result.get("duplicates_skipped", 0),
        status=result["status"],
        message=f"Indexed {len(files)} file(s).",
    )

@app.delete("/memory/{session_id}", tags=["Session"], status_code=status.HTTP_204_NO_CONTENT)
async def clear_session_memory(session_id: str, request: Request) -> None:
    eng = _engine_or_503(request)
    eng.clear_memory(session_id)

@app.delete("/memory", tags=["Session"], status_code=status.HTTP_204_NO_CONTENT)
async def clear_all_memory(request: Request) -> None:
    eng = _engine_or_503(request)
    eng.clear_all_memory()
    logger.info("All session memories cleared.")
