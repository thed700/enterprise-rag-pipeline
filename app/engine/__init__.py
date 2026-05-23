"""
app/engine/__init__.py — AuraRAG v3.6
Re-exports the public API from the pipeline module so that any existing
code importing from `app.engine` continues to work unchanged.
"""

from app.engine.pipeline import (
    RAGEngine,
    build_llm,
    CrossEncoderReranker,
    RerankedRetriever,
    SessionMemoryStore,
    PROVIDER_MODELS,
    SESSION_TTL_MINUTES,
    validate_provider_config,
)

__all__ = [
    "RAGEngine",
    "build_llm",
    "CrossEncoderReranker",
    "RerankedRetriever",
    "SessionMemoryStore",
    "PROVIDER_MODELS",
    "SESSION_TTL_MINUTES",
    "validate_provider_config",
]
