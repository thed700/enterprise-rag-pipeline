"""
models.py - Pydantic Data Schemas
Author: Akmal Raxmatov (github: thed700)
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Schema for a user query."""
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of source docs to return")


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
    version: str = "1.0.0"
