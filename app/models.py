"""
models.py — AuraRAG Pydantic Data Schemas.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field, SecretStr

from app.utils import APP_VERSION

class PromptOverrides(BaseModel):
    rewrite: str = Field(default="", description="Override the rewrite-node system prompt.")
    grade: str = Field(default="", description="Override the grade-node system prompt.")
    generate: str = Field(default="", description="Override the generate-node system prompt.")
    reflect: str = Field(default="", description="Override the reflect-node system prompt.")

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    provider: str = Field(default="OpenAI")
    model: str = Field(default="gpt-4o-mini")
    api_key: SecretStr = Field(default=SecretStr(""))
    session_id: str = Field(default="default", max_length=128)
    system_prompts: PromptOverrides = Field(default_factory=PromptOverrides)

class StreamQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    provider: str = Field(default="OpenAI")
    model: str = Field(default="gpt-4o-mini")
    api_key: SecretStr = Field(default=SecretStr(""))
    session_id: str = Field(default="default", max_length=128)
    system_prompts: PromptOverrides = Field(default_factory=PromptOverrides)

class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument] = Field(default_factory=list)
    chat_history: List[str] = Field(default_factory=list)
    session_id: str = "default"
    pipeline_trace: List[str] = Field(
        default_factory=list,
        description="Ordered list of LangGraph node names executed for this query.",
    )
    graded_chunks: int = Field(default=0)
    reflect_loops: int = Field(default=0)

class IngestResponse(BaseModel):
    chunks_ingested: int
    duplicates_skipped: int = 0
    status: str
    message: str = ""

class HealthResponse(BaseModel):
    status: str
    vector_store: str
    bm25_index: str
    docs_indexed: str
    bm25_docs: str = "0"
    active_sessions: str = "0"
    version: str = APP_VERSION

class ProvidersResponse(BaseModel):
    providers: Dict[str, List[str]]
