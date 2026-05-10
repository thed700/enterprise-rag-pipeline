"""
models.py - Pydantic Data Schemas
Author: Akmal Raxmatov (github: thed700)

FIXES applied (v3.0.0):
  BUG-06: api_key is now typed as SecretStr so it is masked in all
          __repr__ / __str__ output and never appears in log lines.
          Use .get_secret_value() in engine.py to extract the raw string.
  BUG-09: HealthResponse.version updated to 3.0.0.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, SecretStr


class QueryRequest(BaseModel):
    """Schema for a user query."""
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of source docs to return")
    provider: str = Field(default="OpenAI", description="LLM provider name")
    model: str = Field(default="gpt-4o-mini", description="Model identifier")
    # FIX BUG-06: SecretStr masks the key in logs / repr output.
    api_key: SecretStr = Field(default=SecretStr(""), description="User-supplied API key (ephemeral, never logged)")


class SourceDocument(BaseModel):
    """Schema for a retrieved source document."""
    content: str
    metadata: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    """Schema for a RAG query response."""
    answer: str
    sources: List[SourceDocument] = []
    chat_history: List[str] = []


class IngestResponse(BaseModel):
    """Schema for document ingestion response."""
    chunks_ingested: int
    status: str
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
    vector_store: str
    bm25_index: str
    docs_indexed: str
    version: str = "3.0.0"
