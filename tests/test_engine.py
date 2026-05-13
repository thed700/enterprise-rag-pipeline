"""
tests/test_engine.py — AuraRAG v3.1.0
Author: Akmal Raxmatov (github: thed700)

Run: pytest tests/ -v

Changes v3.1.0:
  - All langchain_classic imports replaced with langchain (BUG-A/B/K)
  - TestSessionMemoryStore: per-session isolation, TTL eviction (BUG-P)
  - TestDeduplication: same-file double-upload skips duplicates (BUG-E)
  - TestHealthDocCount: health() uses ChromaDB count, not _all_docs (BUG-I)
  - TestConstants: PROVIDER_MODELS importable from constants.py (BUG-O)
  - HealthResponse version now tracks APP_VERSION from utils (BUG-L)
  - IngestResponse has duplicates_skipped field (BUG-E)
  - QueryRequest has session_id field (BUG-P)
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

# ─────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────

DOCS_A = [
    Document(page_content="LangChain is a framework for LLM applications.", metadata={"source": "a.pdf", "page": 0}),
    Document(page_content="ChromaDB is a vector database for AI workloads.", metadata={"source": "a.pdf", "page": 1}),
    Document(page_content="Hybrid search combines dense and sparse retrieval.", metadata={"source": "a.pdf", "page": 2}),
]

DOCS_B = [
    Document(page_content="AuraRAG supports multi-provider LLM backends.", metadata={"source": "b.pdf", "page": 0}),
]


def _make_engine():
    """Construct a RAGEngine with all heavy dependencies mocked."""
    with patch("app.engine.CrossEncoder") as mock_ce, \
         patch("app.engine.HuggingFaceEmbeddings"), \
         patch("app.engine.chromadb.PersistentClient"), \
         patch("app.engine.Chroma") as mock_chroma_cls, \
         patch("app.engine.RAGEngine._load_bm25_from_disk"), \
         patch("app.engine.RAGEngine._save_bm25_to_disk"):

        mock_ce.return_value = MagicMock()

        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_store = MagicMock()
        mock_store._collection = mock_col
        mock_chroma_cls.return_value = mock_store

        with patch("app.engine.Chroma.from_documents", return_value=mock_store):
            from app.engine import RAGEngine
            engine = RAGEngine()

    return engine, mock_store


# ─────────────────────────────────────────────
# BUG-A/B/K regression — import chain
# ─────────────────────────────────────────────

class TestImports:
    def test_engine_imports_without_error(self):
        """BUG-A/B/K: engine.py must import cleanly (no langchain_classic)."""
        import importlib
        import app.engine as eng
        importlib.reload(eng)   # force re-import; would raise if langchain_classic referenced

    def test_constants_importable_independently(self):
        """BUG-O: constants.py must be importable without heavy ML deps."""
        from app.constants import PROVIDER_MODELS, validate_provider_config, friendly_model_label
        assert "OpenAI" in PROVIDER_MODELS
        assert "Anthropic" in PROVIDER_MODELS

    def test_validate_provider_config_in_constants(self):
        from app.constants import validate_provider_config
        ok, _ = validate_provider_config("Local (Ollama)", "")
        assert ok is True
        ok2, _ = validate_provider_config("OpenAI", "short")
        assert ok2 is False


# ─────────────────────────────────────────────
# CrossEncoderReranker
# ─────────────────────────────────────────────

class TestReranker:
    @patch("app.engine.CrossEncoder")
    def test_returns_top_k(self, mock_ce):
        from app.engine import CrossEncoderReranker
        mock_ce.return_value.predict.return_value = [0.9, 0.5, 0.1]
        r = CrossEncoderReranker()
        result = r.rerank("query", DOCS_A, top_k=2)
        assert len(result) == 2

    @patch("app.engine.CrossEncoder")
    def test_sorted_by_score(self, mock_ce):
        from app.engine import CrossEncoderReranker
        mock_ce.return_value.predict.return_value = [0.1, 0.95, 0.4]
        r = CrossEncoderReranker()
        result = r.rerank("ChromaDB", DOCS_A, top_k=3)
        assert "ChromaDB" in result[0].page_content

    @patch("app.engine.CrossEncoder")
    def test_empty_input(self, mock_ce):
        from app.engine import CrossEncoderReranker
        r = CrossEncoderReranker()
        assert r.rerank("q", [], top_k=5) == []


# ─────────────────────────────────────────────
# SessionMemoryStore  (BUG-P)
# ─────────────────────────────────────────────

class TestSessionMemoryStore:
    def test_separate_sessions_are_independent(self):
        from app.engine import SessionMemoryStore
        store = SessionMemoryStore(window_k=5)
        m1 = store.get("alice")
        m2 = store.get("bob")
        assert m1 is not m2, "Different session_ids must return different memory objects."

    def test_same_session_returns_same_object(self):
        from app.engine import SessionMemoryStore
        store = SessionMemoryStore(window_k=5)
        m1 = store.get("alice")
        m2 = store.get("alice")
        assert m1 is m2

    def test_clear_specific_session(self):
        from app.engine import SessionMemoryStore
        store = SessionMemoryStore(window_k=5)
        store.get("alice")
        store.get("bob")
        store.clear("alice")
        # bob should still exist
        assert store.active_sessions == 1

    def test_ttl_eviction(self):
        from app.engine import SessionMemoryStore, SESSION_TTL_MINUTES
        store = SessionMemoryStore(window_k=5)
        store.get("old_session")
        # Backdate last_access to simulate stale session
        store._last_access["old_session"] = time.monotonic() - (SESSION_TTL_MINUTES * 60 + 1)
        store._evict_stale()
        assert store.active_sessions == 0


# ─────────────────────────────────────────────
# Deduplication  (BUG-E)
# ─────────────────────────────────────────────

class TestDeduplication:
    def test_second_ingest_of_same_file_skips_duplicates(self):
        engine, _ = _make_engine()
        r1 = engine.ingest_documents(DOCS_A)
        first_count = r1["chunks_ingested"]

        r2 = engine.ingest_documents(DOCS_A)   # exact same docs
        assert r2["chunks_ingested"] == 0, "Duplicate chunks must be skipped."
        assert r2["duplicates_skipped"] == first_count

    def test_new_docs_still_ingested_after_duplicates(self):
        engine, _ = _make_engine()
        engine.ingest_documents(DOCS_A)
        r = engine.ingest_documents(DOCS_A + DOCS_B)
        assert r["chunks_ingested"] > 0, "New content in DOCS_B must be ingested."


# ─────────────────────────────────────────────
# Ingestion basics
# ─────────────────────────────────────────────

class TestIngestion:
    def test_returns_success_status(self):
        engine, _ = _make_engine()
        result = engine.ingest_documents(DOCS_A)
        assert result["status"] == "success"

    def test_builds_bm25_index(self):
        engine, _ = _make_engine()
        engine.ingest_documents(DOCS_A)
        assert engine._bm25_retriever is not None

    def test_accumulates_corpus(self):
        """BUG-03 regression: corpus must grow across ingests."""
        engine, _ = _make_engine()
        engine.ingest_documents(DOCS_A)
        c1 = len(engine._all_docs)
        engine.ingest_documents(DOCS_B)
        assert len(engine._all_docs) > c1


# ─────────────────────────────────────────────
# Health  (BUG-I)
# ─────────────────────────────────────────────

class TestHealth:
    def test_empty_before_ingest(self):
        engine, _ = _make_engine()
        h = engine.health()
        assert h["vector_store"] == "empty"
        assert h["bm25_index"] == "empty"

    def test_uses_chroma_count_not_all_docs(self):
        """BUG-I: docs_indexed must come from ChromaDB, not len(_all_docs)."""
        engine, mock_store = _make_engine()
        # Simulate ChromaDB having 500 docs (e.g. after restart, _all_docs may be empty)
        mock_store._collection.count.return_value = 500
        engine.vector_store = mock_store   # mark as ready
        h = engine.health()
        assert h["docs_indexed"] == "500", (
            "BUG-I: health() must read ChromaDB count, not _all_docs length."
        )

    def test_active_sessions_reported(self):
        engine, _ = _make_engine()
        engine._session_store.get("u1")
        engine._session_store.get("u2")
        h = engine.health()
        assert h["active_sessions"] == "2"


# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────

class TestSchemas:
    def test_query_request_has_session_id(self):
        """BUG-P: QueryRequest must carry session_id."""
        from app.models import QueryRequest
        req = QueryRequest(question="Hello?", session_id="sess-abc")
        assert req.session_id == "sess-abc"

    def test_query_request_default_session_id(self):
        from app.models import QueryRequest
        req = QueryRequest(question="Hello?")
        assert req.session_id == "default"

    def test_ingest_response_has_duplicates_skipped(self):
        """BUG-E: IngestResponse must expose duplicates_skipped."""
        from app.models import IngestResponse
        r = IngestResponse(chunks_ingested=10, duplicates_skipped=3, status="success")
        assert r.duplicates_skipped == 3

    def test_health_version_matches_app_version(self):
        """BUG-L: HealthResponse.version must equal APP_VERSION."""
        from app.models import HealthResponse
        from app.utils import APP_VERSION
        r = HealthResponse(status="ok", vector_store="ready", bm25_index="ready", docs_indexed="0")
        assert r.version == APP_VERSION, (
            f"BUG-L: version mismatch — got {r.version!r}, expected {APP_VERSION!r}."
        )

    def test_api_key_is_secret_str(self):
        """BUG-06: api_key must be masked in repr."""
        from app.models import QueryRequest
        req = QueryRequest(question="Hi", api_key="sk-supersecret")
        assert "sk-supersecret" not in str(req.api_key)
        assert req.api_key.get_secret_value() == "sk-supersecret"

    def test_providers_response(self):
        from app.models import ProvidersResponse
        from app.constants import PROVIDER_MODELS
        r = ProvidersResponse(providers=PROVIDER_MODELS)
        assert "OpenAI" in r.providers
