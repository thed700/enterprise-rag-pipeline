"""
app/engine.py — AuraRAG v3.6 legacy shim.

This file exists for backward compatibility only. All engine logic has
lived in app/engine/pipeline.py since v3.4. Importing from app.engine
still works correctly via app/engine/__init__.py.
"""
from app.engine.pipeline import (  # noqa: F401  (re-export for old imports)
    RAGEngine,
    build_llm,
    CrossEncoderReranker,
    RerankedRetriever,
    SessionMemoryStore,
    PROVIDER_MODELS,
    SESSION_TTL_MINUTES,
    validate_provider_config,
)
