"""
models.py — Pydantic Data Schemas v3.2.0
Author: Akmal Raxmatov (github: thed700)

Changes v3.2.0:
  BUG-S: top_k was declared on QueryRequest but never forwarded to engine.query()
          (the engine always used its default of 5). Now threaded through to
          engine.query() and engine.stream_query() via the API layer.
  BUG-T: IngestResponse.message typed as Optional[str] but was always populated.
          Changed to str with a sensible default to reflect actual behaviour.
  NEW:   StreamQueryRequest gains top_k field to match QueryRequest.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, SecretStr

from app.utils import APP_VERSION   # single source of truth (BUG-L)


class QueryRequest(BaseModel):
    """Schema for a /query request."""
    question:   str       = Field(..., min_length=1, max_length=2000)
    top_k:      int       = Field(default=5, ge=1, le=20)
    provider:   str       = Field(default="OpenAI")
    model:      str       = Field(default="gpt-4o-mini")
    api_key:    SecretStr = Field(default=SecretStr(""))  # never logged
    session_id: str       = Field(default="default", max_length=128)


class StreamQueryRequest(BaseModel):
    """Schema for the /query/stream SSE endpoint."""
    question:   str       = Field(..., min_length=1, max_length=2000)
    # BUG-S fix: top_k added so streaming respects the caller's preference
    top_k:      int       = Field(default=5, ge=1, le=20)
    provider:   str       = Field(default="OpenAI")
    model:      str       = Field(default="gpt-4o-mini")
    api_key:    SecretStr = Field(default=SecretStr(""))
    session_id: str       = Field(default="default", max_length=128)


class SourceDocument(BaseModel):
    content:  str
    metadata: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    answer:       str
    sources:      List[SourceDocument] = []
    chat_history: List[str] = []
    session_id:   str = "default"


class IngestResponse(BaseModel):
    chunks_ingested:    int
    duplicates_skipped: int = 0
    status:             str
    # BUG-T fix: always populated — changed from Optional[str] to str
    message:            str = ""


class HealthResponse(BaseModel):
    status:          str
    vector_store:    str
    bm25_index:      str
    docs_indexed:    str
    active_sessions: str = "0"
    version:         str = APP_VERSION   # single source of truth (BUG-L)


class ProvidersResponse(BaseModel):
    """Response for GET /providers."""
    providers: Dict[str, List[str]]
