"""
models.py — AuraRAG v3.4 Pydantic Data Schemas
Author: Akmal Raxmatov (github: thed700)

Changes v3.4:
  - QueryResponse extended with three observability fields populated by the
    LangGraph pipeline:
      pipeline_trace: List[str] — ordered node names executed per query.
      graded_chunks:  int       — chunks that passed the Document Grader.
      reflect_loops:  int       — self-correction loops performed.
  - HealthResponse.version reflects APP_VERSION = "3.4".

Retained from v3.3:
  BUG-AD: HealthResponse includes bm25_docs field (was silently dropped).
  BUG-T:  IngestResponse.message typed as str (not Optional[str]).
  BUG-S:  top_k present on QueryRequest / StreamQueryRequest.
  BUG-L:  APP_VERSION imported from utils — single source of truth.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field, SecretStr

from app.utils import APP_VERSION  # single source of truth (BUG-L)


class QueryRequest(BaseModel):
    """Schema for a POST /query request (non-streaming)."""

    question:   str       = Field(..., min_length=1, max_length=2000)
    top_k:      int       = Field(default=5, ge=1, le=20)
    provider:   str       = Field(default="OpenAI")
    model:      str       = Field(default="gpt-4o-mini")
    api_key:    SecretStr = Field(default=SecretStr(""))  # never logged
    session_id: str       = Field(default="default", max_length=128)


class StreamQueryRequest(BaseModel):
    """Schema for POST /query/stream (SSE)."""

    question:   str       = Field(..., min_length=1, max_length=2000)
    # BUG-S fix (v3.2.0): top_k forwarded so streaming respects caller preference
    top_k:      int       = Field(default=5, ge=1, le=20)
    provider:   str       = Field(default="OpenAI")
    model:      str       = Field(default="gpt-4o-mini")
    api_key:    SecretStr = Field(default=SecretStr(""))
    session_id: str       = Field(default="default", max_length=128)


class SourceDocument(BaseModel):
    content:  str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer:       str
    sources:      List[SourceDocument] = Field(default_factory=list)
    chat_history: List[str]            = Field(default_factory=list)
    session_id:   str                  = "default"

    # v3.4 observability — populated by the LangGraph pipeline.
    # Clients on older v3.3 contracts can ignore these fields safely.
    pipeline_trace: List[str] = Field(
        default_factory=list,
        description="Ordered list of LangGraph node names executed for this query.",
    )
    graded_chunks: int = Field(
        default=0,
        description="Number of retrieved chunks that passed the Document Grader.",
    )
    reflect_loops: int = Field(
        default=0,
        description="Number of Reflect/Self-Correction loops performed.",
    )


class IngestResponse(BaseModel):
    chunks_ingested:    int
    duplicates_skipped: int = 0
    status:             str
    # BUG-T fix (v3.2.0): always populated — str, not Optional[str]
    message:            str = ""


class HealthResponse(BaseModel):
    status:          str
    vector_store:    str
    bm25_index:      str
    docs_indexed:    str
    # BUG-AD fix (v3.3): bm25_docs was returned by engine.health() but was
    # missing here, so FastAPI silently dropped it from every /health response.
    bm25_docs:       str = "0"
    active_sessions: str = "0"
    version:         str = APP_VERSION  # single source of truth (BUG-L)


class ProvidersResponse(BaseModel):
    """Response schema for GET /providers."""

    providers: Dict[str, List[str]]
