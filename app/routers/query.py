"""
routers/query.py — AuraRAG v3.4 Query Router
Author: Akmal Raxmatov (github: thed700)

Changes v3.4:
  - POST /query calls engine.aquery() directly (fully async — no to_thread()
    wrapper needed since the LangGraph pipeline is async-first throughout).
  - POST /query/stream uses engine.stream_query() backed by LangGraph's
    native astream() — no bridge thread, no AsyncIteratorCallbackHandler
    data race (BUG-AH fix).
  - QueryResponse now includes pipeline_trace, graded_chunks, reflect_loops
    for observability into the agentic pipeline.
  - SSE stream emits a final "meta" event after [DONE] with per-request
    observability data (session_id, token_count, provider, model, top_k).
  - Engine accessed via request.app.state.engine (no module-level global).

Retained from v3.3:
  BUG-S: top_k forwarded from QueryRequest / StreamQueryRequest into engine.
  BUG-V: chain exceptions propagated to SSE error frames.
  Rate limiting (30/minute query) via slowapi.
"""

import json
import logging
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.engine import RAGEngine
from app.models import (
    QueryRequest,
    QueryResponse,
    SourceDocument,
    StreamQueryRequest,
)
from app.utils import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

router  = APIRouter(tags=["Retrieval"])
limiter = Limiter(key_func=get_remote_address)


# ─────────────────────────────────────────────
# DEPENDENCY — Engine access
# ─────────────────────────────────────────────

def _get_engine(request: Request) -> RAGEngine:
    """FastAPI dependency: retrieves the shared RAGEngine from app.state."""
    eng: Optional[RAGEngine] = getattr(request.app.state, "engine", None)
    if eng is None:
        raise HTTPException(status_code=503, detail="Engine not initialised.")
    return eng


# ─────────────────────────────────────────────
# POST /query  (non-streaming)
# ─────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="LangGraph RAG Query",
    description=(
        "Full agentic RAG query via LangGraph: "
        "Query Rewrite → Hybrid Retrieve → Document Grade → Generate → [Reflect]. "
        "Returns the grounded answer, source chunks, session history, "
        "and pipeline observability fields (pipeline_trace, graded_chunks, reflect_loops)."
    ),
)
@limiter.limit(settings.RATE_LIMIT_QUERY)
async def query_rag(
    request: Request,
    body:    QueryRequest,
    engine:  RAGEngine = Depends(_get_engine),
) -> QueryResponse:
    """
    LangGraph agentic RAG query (non-streaming).

    The LangGraph pipeline is fully async — no asyncio.to_thread() wrapper
    needed.  top_k is forwarded end-to-end (BUG-S fix).
    """
    logger.info("[%s] /query: '%s'", body.session_id, body.question[:60])

    result = await engine.aquery(
        question=body.question,
        provider=body.provider,
        model=body.model,
        api_key=body.api_key.get_secret_value(),
        session_id=body.session_id,
        top_k=body.top_k,  # BUG-S: forwarded
    )

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceDocument(**s) for s in result["sources"]],
        chat_history=result["chat_history"],
        session_id=result["session_id"],
        pipeline_trace=result.get("pipeline_trace", []),
        graded_chunks=result.get("graded_chunks", 0),
        reflect_loops=result.get("reflect_loops", 0),
    )


# ─────────────────────────────────────────────
# POST /query/stream  (SSE streaming)
# ─────────────────────────────────────────────

@router.post(
    "/query/stream",
    summary="LangGraph SSE Streaming Query",
    description=(
        "True Server-Sent Events streaming query via LangGraph astream(). "
        "Tokens are streamed from the Generate node as they arrive. "
        "Event types:\n"
        "  {\"token\": \"...\"} — during token generation\n"
        "  {\"error\": \"...\"} — on exception\n"
        "  [DONE]              — stream termination sentinel\n"
        "  {\"meta\": {...}}   — pipeline observability data after [DONE]"
    ),
)
@limiter.limit(settings.RATE_LIMIT_QUERY)
async def query_stream(
    request: Request,
    body:    StreamQueryRequest,
    engine:  RAGEngine = Depends(_get_engine),
) -> StreamingResponse:
    """
    SSE streaming via LangGraph's native astream() API.

    BUG-AH fix (v3.4): no worker thread, no AsyncIteratorCallbackHandler,
    no data race on asyncio.Queue at shutdown.  All I/O stays on the event
    loop from start to finish.

    BUG-V fix (v3.2.0): exceptions are caught and emitted as SSE error frames.
    BUG-S fix (v3.2.0): top_k forwarded to the engine.
    """
    logger.info("[%s] /query/stream: '%s'", body.session_id, body.question[:60])

    async def event_generator() -> AsyncGenerator[str, None]:
        token_count = 0
        try:
            async for token in engine.stream_query(
                question=body.question,
                provider=body.provider,
                model=body.model,
                api_key=body.api_key.get_secret_value(),
                session_id=body.session_id,
                top_k=body.top_k,  # BUG-S: forwarded
            ):
                token_count += 1
                yield f"data: {json.dumps({'token': token})}\n\n"

        except Exception as exc:
            logger.exception("[%s] Stream error", body.session_id)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        finally:
            # Termination sentinel
            yield "data: [DONE]\n\n"
            # Observability meta event — always emitted so clients can log it
            meta = json.dumps({
                "meta": {
                    "session_id":  body.session_id,
                    "token_count": token_count,
                    "provider":    body.provider,
                    "model":       body.model,
                    "top_k":       body.top_k,
                }
            })
            yield f"data: {meta}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",  # reverse-proxy compatibility
        },
    )
