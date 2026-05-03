"""
main.py - FastAPI Backend Server
Author: Akmal Raxmatov (github: thed700)

Endpoints:
  POST /ingest      — Upload and index documents
  POST /query       — Query the RAG pipeline
  DELETE /memory    — Clear conversation memory
  GET  /health      — Engine health check
"""

import logging
import tempfile
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.document_loaders import PyPDFLoader, TextLoader

from app.engine import RAGEngine
from app.models import QueryRequest, QueryResponse, IngestResponse, HealthResponse, SourceDocument
from app.utils import setup_logging, get_settings

# ── Bootstrap ───────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

# Shared engine instance (singleton)
engine: RAGEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Starting Enterprise RAG API...")
    engine = RAGEngine()
    yield
    logger.info("Shutting down Enterprise RAG API.")


# ── FastAPI App ──────────────────────────────
app = FastAPI(
    title="Enterprise RAG Pipeline API",
    description="Hybrid Search + Cross-Encoder Re-ranking RAG system by Akmal Raxmatov",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Returns engine health status."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised.")
    h = engine.health()
    return HealthResponse(status="ok", **h)


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
    Upload and index documents into the RAG pipeline.
    Supported formats: PDF, TXT
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready.")

    from langchain.schema import Document

    all_docs = []
    for upload in files:
        suffix = os.path.splitext(upload.filename or "")[-1].lower()
        if suffix not in (".pdf", ".txt"):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {suffix}. Use PDF or TXT.",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await upload.read())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = upload.filename
            all_docs.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from '{upload.filename}'.")
        finally:
            os.unlink(tmp_path)

    if not all_docs:
        raise HTTPException(status_code=400, detail="No content extracted from files.")

    result = engine.ingest_documents(all_docs)
    return IngestResponse(
        chunks_ingested=result["chunks_ingested"],
        status=result["status"],
        message=f"Successfully indexed {len(files)} file(s).",
    )


@app.post("/query", response_model=QueryResponse, tags=["Retrieval"])
async def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Query the RAG pipeline.
    Runs Hybrid Search → Cross-Encoder Re-ranking → LLM generation.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready.")

    logger.info(f"Query received: '{request.question[:60]}...'")
    result = engine.query(request.question)

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
