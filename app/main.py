"""
main.py — FastAPI Backend v3.2.0
Author: Akmal Raxmatov (github: thed700)

Bug fixes in v3.2.0:
  BUG-S: top_k now forwarded from QueryRequest / StreamQueryRequest into
          engine.query() and engine.stream_query() — was silently dropped.
  BUG-Q: setup_logging() now reads Settings.LOG_LEVEL (was always INFO).
  BUG-W: TextLoader called with encoding="utf-8" and autodetect_encoding=True
          to handle non-UTF-8 .txt uploads without raising UnicodeDecodeError.

Retained from v3.1.0:
  BUG-F: MAX_UPLOAD_MB enforced — file size checked before reading into memory.
  BUG-G: upload.read() replaced with chunked streaming write to temp file.
  BUG-H: slowapi rate limiting on /query (30/min) and /ingest (10/min).
  BUG-O: GET /providers returns PROVIDER_MODELS so the UI fetches it over HTTP.
  BUG-P: session_id threaded through /query and /query/stream.
"""

import asyncio
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from app.constants import PROVIDER_MODELS
from app.engine import RAGEngine
from app.models import (
    IngestResponse,
    HealthResponse,
    ProvidersResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    StreamQueryRequest,
)
from app.utils import APP_VERSION, get_settings, setup_logging

# BUG-Q fix: setup_logging() now reads LOG_LEVEL from Settings
setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────
# RATE LIMITER
# ─────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ─────────────────────────────────────────────
# APP LIFECYCLE
# ─────────────────────────────────────────────

engine: RAGEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("=" * 60)
    logger.info(f"  AuraRAG — Advanced Unified Retrieval Architecture  v{APP_VERSION}")
    logger.info("=" * 60)
    engine = RAGEngine()
    yield
    logger.info("AuraRAG API shutting down.")


app = FastAPI(
    title="AuraRAG API",
    description=(
        "Advanced Unified Retrieval Architecture v3.2.0. "
        "Hybrid Search + Cross-Encoder Re-ranking. "
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
# HELPERS
# ─────────────────────────────────────────────

_CHUNK_SIZE = 1024 * 256  # 256 KB streaming write chunks


async def _stream_upload_to_tmp(upload: UploadFile, suffix: str) -> str:
    """
    Stream-write an uploaded file to a named temp file in 256 KB chunks.
    Enforces MAX_UPLOAD_MB limit during streaming.
    Returns the temp file path.
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


def _engine_or_503() -> RAGEngine:
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised.")
    return engine


# ─────────────────────────────────────────────
# ROUTES — System
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    eng = _engine_or_503()
    return HealthResponse(status="ok", **eng.health())


@app.get("/providers", response_model=ProvidersResponse, tags=["System"])
async def list_providers() -> ProvidersResponse:
    """Returns the provider → model map."""
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
    files: List[UploadFile] = File(...),
) -> IngestResponse:
    """Upload and index PDF or TXT documents."""
    eng = _engine_or_503()

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
                # BUG-W fix: explicit UTF-8 + autodetect fallback avoids
                # UnicodeDecodeError on non-UTF-8 text files in C-locale containers.
                loader = TextLoader(tmp_path, encoding="utf-8", autodetect_encoding=True)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = upload.filename
            all_docs.extend(docs)
            logger.info(f"Loaded {len(docs)} page(s) from '{upload.filename}'.")
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


# ─────────────────────────────────────────────
# ROUTES — Query
# ─────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["Retrieval"])
@limiter.limit(settings.RATE_LIMIT_QUERY)
async def query_rag(request: Request, body: QueryRequest) -> QueryResponse:
    """
    RAG query: Hybrid Search → Re-rank → LLM → Answer.
    Pass session_id to maintain per-user conversation history.
    """
    eng = _engine_or_503()
    logger.info(f"[{body.session_id}] Query: '{body.question[:60]}'")

    result = await asyncio.to_thread(
        eng.query,
        question=body.question,
        provider=body.provider,
        model=body.model,
        api_key=body.api_key.get_secret_value(),
        session_id=body.session_id,
        top_k=body.top_k,          # BUG-S fix: forwarded
    )

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceDocument(**s) for s in result["sources"]],
        chat_history=result["chat_history"],
        session_id=result["session_id"],
    )


@app.post("/query/stream", tags=["Retrieval"])
@limiter.limit(settings.RATE_LIMIT_QUERY)
async def query_stream(request: Request, body: StreamQueryRequest) -> StreamingResponse:
    """
    True SSE streaming query. Returns tokens as they arrive from the LLM.
    BUG-V fix: chain exceptions now propagate to the SSE error frame.
    """
    eng = _engine_or_503()
    logger.info(f"[{body.session_id}] Stream query: '{body.question[:60]}'")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for token in eng.stream_query(
                question=body.question,
                provider=body.provider,
                model=body.model,
                api_key=body.api_key.get_secret_value(),
                session_id=body.session_id,
                top_k=body.top_k,   # BUG-S fix: forwarded
            ):
                payload = json.dumps({"token": token})
                yield f"data: {payload}\n\n"
        except Exception as e:
            logger.exception("Stream error")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────
# ROUTES — Session Memory
# ─────────────────────────────────────────────

@app.delete(
    "/memory/{session_id}",
    tags=["Session"],
    status_code=status.HTTP_204_NO_CONTENT,
)
async def clear_session_memory(session_id: str) -> None:
    """Clear conversation history for a specific session."""
    eng = _engine_or_503()
    eng.clear_memory(session_id)


@app.delete(
    "/memory",
    tags=["Session"],
    status_code=status.HTTP_204_NO_CONTENT,
)
async def clear_all_memory() -> None:
    """Clear ALL session memories (admin operation)."""
    eng = _engine_or_503()
    eng.clear_all_memory()
    logger.info("All session memories cleared.")
