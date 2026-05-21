"""
routers/query.py — AuraRAG Query Router.
"""

from __future__ import annotations

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

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(tags=["Retrieval"])
limiter = Limiter(key_func=get_remote_address)

def _get_engine(request: Request) -> RAGEngine:
    eng: Optional[RAGEngine] = getattr(request.app.state, "engine", None)
    if eng is None:
        raise HTTPException(status_code=503, detail="Engine not initialised.")
    return eng

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="LangGraph RAG Query",
    description=(
        "Full agentic RAG query via LangGraph: Query Rewrite → Hybrid Retrieve → "
        "Document Grade → Generate → [Reflect]."
    ),
)
@limiter.limit(settings.RATE_LIMIT_QUERY)
async def query_rag(
    request: Request,
    body: QueryRequest,
    engine: RAGEngine = Depends(_get_engine),
) -> QueryResponse:
    logger.info("[%s] /query: '%s'", body.session_id, body.question[:60])

    result = await engine.aquery(
        question=body.question,
        provider=body.provider,
        model=body.model,
        api_key=body.api_key.get_secret_value(),
        session_id=body.session_id,
        top_k=body.top_k,
        system_prompts=body.system_prompts.model_dump(exclude_none=True),
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

@router.post(
    "/query/stream",
    summary="LangGraph SSE Streaming Query",
    description=(
        "True Server-Sent Events streaming query via LangGraph astream()."
    ),
)
@limiter.limit(settings.RATE_LIMIT_QUERY)
async def query_stream(
    request: Request,
    body: StreamQueryRequest,
    engine: RAGEngine = Depends(_get_engine),
) -> StreamingResponse:
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
                top_k=body.top_k,
                system_prompts=body.system_prompts.model_dump(exclude_none=True),
            ):
                token_count += 1
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as exc:
            logger.exception("[%s] Stream error", body.session_id)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            # BUG-AM fix (v3.5): the meta frame was emitted AFTER the [DONE]
            # sentinel.  Clients that stop reading at [DONE] (which is the
            # standard SSE termination signal) never received the metadata.
            # Fix: emit meta first, then [DONE] so compliant clients see both.
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
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
