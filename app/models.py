"""
models.py — Pydantic Data Schemas v3.1.0
Author: Akmal Raxmatov (github: thed700)

Changes v3.1.0:
  BUG-L: HealthResponse.version now imports APP_VERSION from utils instead of
          hardcoding the literal string — single source of truth.
  BUG-P: QueryRequest gains optional session_id field for per-session memory.
  NEW:   IngestResponse gains duplicates_skipped field (BUG-E).
  NEW:   ProvidersResponse for the /providers endpoint.
  NEW:   StreamQueryRequest mirrors QueryRequest for the SSE stream endpoint.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, SecretStr

from app.utils import APP_VERSION   # FIX BUG-L: single source of truth


class QueryRequest(BaseModel):
    """Schema for a /query request."""
    question:   str       = Field(..., min_length=1, max_length=2000)
    top_k:      int       = Field(default=5, ge=1, le=20)
    provider:   str       = Field(default="OpenAI")
    model:      str       = Field(default="gpt-4o-mini")
    api_key:    SecretStr = Field(default=SecretStr(""))  # BUG-06: never logged
    # FIX BUG-P: per-session memory isolation
    session_id: str       = Field(default="default", max_length=128)


class StreamQueryRequest(BaseModel):
    """Schema for the /query/stream SSE endpoint."""
    question:   str       = Field(..., min_length=1, max_length=2000)
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
    chunks_ingested:   int
    duplicates_skipped: int = 0   # BUG-E: surfaced in API response
    status:            str
    message:           Optional[str] = None


class HealthResponse(BaseModel):
    status:          str
    vector_store:    str
    bm25_index:      str
    docs_indexed:    str
    active_sessions: str = "0"
    version:         str = APP_VERSION   # FIX BUG-L


class ProvidersResponse(BaseModel):
    """Response for GET /providers."""
    providers: Dict[str, List[str]]
