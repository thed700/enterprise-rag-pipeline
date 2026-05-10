"""
main.py - FastAPI Backend Server
Author: Akmal Raxmatov (github: thed700)

FIXES applied (v3.0.0):
  BUG-02: engine.query() is a blocking synchronous call. Running it directly
          inside an async FastAPI route handler freezes the event loop for the
          entire duration of the LLM call, stalling all other in-flight requests.
          Fixed: wrapped in asyncio.to_thread() so it runs in a thread-pool worker.
  BUG-05: Temporary file path could be undefined if NamedTemporaryFile raised
          before the assignment, causing a NameError in the finally block and
          masking the original exception. Fixed with a None-guarded cleanup.
  BUG-06: api_key is now SecretStr in QueryRequest; .get_secret_value() used
          when forwarding to engine.query() to avoid key leaking into logs.
  BUG-09: CORS allow_origins=['*'] + allow_credentials=True is both a spec
          violation and a security hole. Replaced with an explicit origins list
          read from the ALLOWED_ORIGINS environment variable.
"""

import asyncio
import logging
import tempfile
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from app.engine import RAGEngine
from app.models import QueryRequest, QueryResponse, IngestResponse, HealthResponse, SourceDocument
from app.utils import setup_logging, get_settings, APP_VERSION

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

engine: RAGEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("=" * 60)
    logger.info(f"  AuraRAG — Advanced Unified Retrieval Architecture  v{APP_VERSION}")
    logger.info("=" * 60)
    engine = RAGEngine()
    yield
    logger.info("Shutting down AuraRAG API.")


app = FastAPI(
    title="AuraRAG API",
    description=(
        "Advanced Unified Retrieval Architecture. "
        "Hybrid Search + Cross-Encoder Re-ranking. "
        "LLM-agnostic: OpenAI · Anthropic · Google Gemini · Ollama."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

# FIX BUG-09: explicit allowed origins (never wildcard + credentials).
# Set ALLOWED_ORIGINS=https://your-domain.com in .env for production.
_raw_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501",
)
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Routes ───────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Returns engine health status."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised.")
    h = engine.health()
    return HealthResponse(status="ok", version=APP_VERSION, **h)


@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestion"],
)
async def ingest_documents(
    files: List[UploadFile] = File(..., description="PDF or TXT files to index"),
) -> IngestResponse:
    """
    Upload and index documents into the AuraRAG pipeline.
    Supported formats: PDF, TXT
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready.")

    all_docs: List[Document] = []
    for upload in files:
        suffix = os.path.splitext(upload.filename or "")[-1].lower()
        if suffix not in (".pdf", ".txt"):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: '{suffix}'. Use PDF or TXT.",
            )

        # FIX BUG-05: guard tmp_path so finally block never raises NameError.
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await upload.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = upload.filename
            all_docs.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from '{upload.filename}'.")
        finally:
            # FIX BUG-05: only unlink if the path was successfully assigned.
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    if not all_docs:
        raise HTTPException(status_code=400, detail="No content extracted from files.")

    # Ingestion is CPU-bound; run in thread-pool to avoid blocking the event loop.
    result = await asyncio.to_thread(engine.ingest_documents, all_docs)
    return IngestResponse(
        chunks_ingested=result["chunks_ingested"],
        status=result["status"],
        message=f"Successfully indexed {len(files)} file(s).",
    )


@app.post("/query", response_model=QueryResponse, tags=["Retrieval"])
async def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Query the AuraRAG pipeline.
    Runs Hybrid Search -> Cross-Encoder Re-ranking -> LLM generation.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready.")

    logger.info(f"Query received: '{request.question[:60]}...'")

    # FIX BUG-02: engine.query() is synchronous (LangChain chain.invoke).
    # asyncio.to_thread() offloads it to a thread-pool worker so the event
    # loop remains unblocked for concurrent requests.
    # FIX BUG-06: extract raw key from SecretStr — never log the object directly.
    result = await asyncio.to_thread(
        engine.query,
        question=request.question,
        provider=request.provider,
        model=request.model,
        api_key=request.api_key.get_secret_value(),
    )

    sources = [SourceDocument(**s) for s in result["sources"]]
    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        chat_history=result["chat_history"],
    )


@app.delete("/memory", tags=["Session"], status_code=status.HTTP_204_NO_CONTENT)
async def clear_memory() -> None:
    """Clear conversation memory to start a fresh session."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready.")
    engine.clear_memory()
    logger.info("Conversation memory cleared via API.")
