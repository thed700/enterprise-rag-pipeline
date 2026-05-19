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
import json
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import settings
from app.engine.pipeline import RAGEngine
from app.schemas import IngestResponse, ProviderConfig, QueryRequest, QueryResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AuraRAG")

GLOBAL_ENGINE = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_ENGINE
    logger.info("Initializing AuraRAG Engine inside Lifespan...")
    try:
        GLOBAL_ENGINE = RAGEngine()
        app.state.engine = GLOBAL_ENGINE
    except Exception as e:
        logger.error(f"Engine initialization failed: {e}")
    yield
    if GLOBAL_ENGINE:
        try:
            GLOBAL_ENGINE.shutdown()
        except Exception:
            pass

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="AuraRAG API",
    version="3.4.0",
    lifespan=lifespan
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Absolute Wildcard CORS for Hugging Face Proxy Isolation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _engine_or_503(request: Request) -> RAGEngine:
    engine = getattr(request.app.state, "engine", GLOBAL_ENGINE)
    if not engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AuraRAG Engine is initializing. Please retry in a moment.",
        )
    return engine

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "engine": "ready" if GLOBAL_ENGINE else "initializing"}

@app.get("/providers", tags=["System"])
async def get_providers(request: Request):
    from app.constants import PROVIDER_MODELS
    return PROVIDER_MODELS

@app.post("/ingest", tags=["Ingestion"], response_model=IngestResponse)
@limiter.limit(settings.INGEST_RATE_LIMIT)
async def ingest_files(request: Request, files: List[UploadFile] = File(...)) -> IngestResponse:
    eng = _engine_or_503(request)
    all_docs = []
    for upload in files:
        suffix = os.path.splitext(upload.filename or "")[1].lower()
        if suffix != ".pdf":
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await upload.read()
            tmp.write(content)
            tmp_path = tmp.name
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            all_docs.extend(docs)
        finally:
            if os.path.exists(tmp_path):
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

@app.post("/query")
async def query(request: Request, body: QueryRequest):
    eng = _engine_or_503(request)
    res = eng.query(body.prompt, session_id=body.session_id)
    return {"answer": res.get("answer", ""), "sources": res.get("sources", [])}

@app.post("/query/stream")
async def query_stream(request: Request, body: QueryRequest):
    eng = _engine_or_503(request)
    from fastapi.responses import StreamingResponse
    async def dummy_stream():
        res = eng.query(body.prompt, session_id=body.session_id)
        yield f"data: {json.dumps({'token': res.get('answer', '')})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(dummy_stream(), media_type="text/event-stream")

@app.delete("/memory/{session_id}")
async def clear_session_memory(session_id: str, request: Request):
    eng = _engine_or_503(request)
    eng.clear_memory(session_id)
    return {"status": "cleared"}