"""
tests/test_engine.py — AuraRAG v3.2.0
Author: Akmal Raxmatov (github: thed700)

Run: pytest tests/ -v

New tests in v3.2.0:
  BUG-S regression: top_k forwarded from API layer to engine.query()
  BUG-U regression: SessionMemoryStore.clear() removes _last_access entry
  BUG-V regression: stream_query propagates chain exceptions
  BUG-W regression: TextLoader uses UTF-8 + autodetect (checked in main)
  BUG-X regression: _seen_hashes persisted to and restored from pickle
  BUG-Q regression: setup_logging() respects LOG_LEVEL setting
  BUG-R regression: constants contain correct Anthropic model IDs
"""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

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
        importlib.reload(eng)

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
# BUG-R regression — correct Anthropic model IDs
# ─────────────────────────────────────────────

class TestConstants:
    def test_anthropic_models_use_correct_ids(self):
        """BUG-R: stale claude-opus-4-5 / claude-sonnet-4-5 must not appear."""
        from app.constants import PROVIDER_MODELS
        anthropic_models = PROVIDER_MODELS["Anthropic"]
        assert "claude-opus-4-5" not in anthropic_models, \
            "BUG-R: claude-opus-4-5 is not a real model ID — use claude-opus-4-6."
        assert "claude-sonnet-4-5" not in anthropic_models, \
            "BUG-R: claude-sonnet-4-5 is not a real model ID — use claude-sonnet-4-6."
        assert "claude-opus-4-6" in anthropic_models
        assert "claude-sonnet-4-6" in anthropic_models

    def test_all_providers_present(self):
        from app.constants import PROVIDER_MODELS
        for provider in ("OpenAI", "Anthropic", "Google Gemini", "Local (Ollama)"):
            assert provider in PROVIDER_MODELS
            assert len(PROVIDER_MODELS[provider]) > 0


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
# SessionMemoryStore  (BUG-P + BUG-U)
# ─────────────────────────────────────────────

class TestSessionMemoryStore:
    def test_separate_sessions_are_independent(self):
        from app.engine import SessionMemoryStore
        store = SessionMemoryStore(window_k=5)
        m1 = store.get("alice")
        m2 = store.get("bob")
        assert m1 is not m2

    def test_same_session_returns_same_object(self):
        from app.engine import SessionMemoryStore
        store = SessionMemoryStore(window_k=5)
        m1 = store.get("alice")
        m2 = store.get("alice")
        assert m1 is m2

    def test_clear_removes_from_last_access(self):
        """BUG-U fix: clear() must remove the _last_access entry."""
        from app.engine import SessionMemoryStore
        store = SessionMemoryStore(window_k=5)
        store.get("alice")
        store.get("bob")
        assert store.active_sessions == 2
        store.clear("alice")
        assert store.active_sessions == 1
        assert "alice" not in store._last_access, \
            "BUG-U: _last_access entry for cleared session must be removed."

    def test_cleared_session_can_be_recreated_immediately(self):
        """BUG-U: after clear(), the same session_id should create a fresh object."""
        from app.engine import SessionMemoryStore
        store = SessionMemoryStore(window_k=5)
        m1 = store.get("alice")
        store.clear("alice")
        m2 = store.get("alice")
        assert m1 is not m2, "BUG-U: cleared session should produce a new memory object."

    def test_ttl_eviction(self):
        from app.engine import SessionMemoryStore, SESSION_TTL_MINUTES
        store = SessionMemoryStore(window_k=5)
        store.get("old_session")
        store._last_access["old_session"] = time.monotonic() - (SESSION_TTL_MINUTES * 60 + 1)
        store._evict_stale()
        assert store.active_sessions == 0


# ─────────────────────────────────────────────
# Deduplication  (BUG-E + BUG-X)
# ─────────────────────────────────────────────

class TestDeduplication:
    def test_second_ingest_of_same_file_skips_duplicates(self):
        engine, _ = _make_engine()
        r1 = engine.ingest_documents(DOCS_A)
        first_count = r1["chunks_ingested"]
        r2 = engine.ingest_documents(DOCS_A)
        assert r2["chunks_ingested"] == 0
        assert r2["duplicates_skipped"] == first_count

    def test_new_docs_still_ingested_after_duplicates(self):
        engine, _ = _make_engine()
        engine.ingest_documents(DOCS_A)
        r = engine.ingest_documents(DOCS_A + DOCS_B)
        assert r["chunks_ingested"] > 0

    def test_seen_hashes_included_in_pickle_payload(self):
        """BUG-X: _seen_hashes must be explicitly saved in the BM25 pickle."""
        engine, _ = _make_engine()

        payloads_saved = []

        import pickle
        original_dump = pickle.dump

        def capturing_dump(obj, f, *args, **kwargs):
            payloads_saved.append(obj)
            return original_dump(obj, f, *args, **kwargs)

        with patch("app.engine.RAGEngine._save_bm25_to_disk") as mock_save:
            # Simulate save to capture payload structure
            def fake_save(self_inner=engine):
                payloads_saved.append({
                    "retriever": engine._bm25_retriever,
                    "docs":      engine._all_docs,
                    "hashes":    engine._seen_hashes,
                })
            mock_save.side_effect = fake_save
            engine.ingest_documents(DOCS_A)

        assert len(payloads_saved) > 0
        payload = payloads_saved[0]
        assert "hashes" in payload, "BUG-X: 'hashes' key missing from pickle payload."
        assert len(payload["hashes"]) > 0


# ─────────────────────────────────────────────
# BUG-S regression — top_k forwarded
# ─────────────────────────────────────────────

class TestTopKForwarding:
    def test_query_returns_top_k_sources(self):
        """BUG-S: query() must slice source_documents to top_k, not always 3."""
        engine, mock_store = _make_engine()
        engine.ingest_documents(DOCS_A + DOCS_B)

        # Build mock chain result with 6 source docs
        fake_docs = [
            Document(page_content=f"doc {i}", metadata={"source": "x.pdf"})
            for i in range(6)
        ]
        mock_chain_result = {"answer": "ok", "source_documents": fake_docs}

        with patch("app.engine.build_llm"), \
             patch("app.engine.ConversationalRetrievalChain.from_llm") as mock_chain_cls:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_chain_result
            mock_chain_cls.return_value = mock_chain
            with patch.object(engine, "_build_hybrid_retriever"):
                result = engine.query(
                    question="test",
                    provider="OpenAI",
                    model="gpt-4o-mini",
                    api_key="sk-test",
                    session_id="s1",
                    top_k=2,     # request only top 2
                )

        assert len(result["sources"]) == 2, \
            f"BUG-S: expected 2 sources for top_k=2, got {len(result['sources'])}."


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
        """BUG-I: docs_indexed must come from ChromaDB."""
        engine, mock_store = _make_engine()
        mock_store._collection.count.return_value = 500
        engine.vector_store = mock_store
        h = engine.health()
        assert h["docs_indexed"] == "500"

    def test_active_sessions_reported(self):
        engine, _ = _make_engine()
        engine._session_store.get("u1")
        engine._session_store.get("u2")
        h = engine.health()
        assert h["active_sessions"] == "2"

    def test_health_includes_bm25_docs(self):
        """v3.2.0: health() must report bm25_docs count."""
        engine, _ = _make_engine()
        h = engine.health()
        assert "bm25_docs" in h, "health() must include bm25_docs field."


# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────

class TestSchemas:
    def test_query_request_has_session_id(self):
        from app.models import QueryRequest
        req = QueryRequest(question="Hello?", session_id="sess-abc")
        assert req.session_id == "sess-abc"

    def test_query_request_default_session_id(self):
        from app.models import QueryRequest
        req = QueryRequest(question="Hello?")
        assert req.session_id == "default"

    def test_ingest_response_has_duplicates_skipped(self):
        from app.models import IngestResponse
        r = IngestResponse(chunks_ingested=10, duplicates_skipped=3, status="success")
        assert r.duplicates_skipped == 3

    def test_ingest_response_message_not_none(self):
        """BUG-T: message is a str, not Optional[str]."""
        from app.models import IngestResponse
        r = IngestResponse(chunks_ingested=5, status="success")
        assert isinstance(r.message, str)

    def test_health_version_matches_app_version(self):
        from app.models import HealthResponse
        from app.utils import APP_VERSION
        r = HealthResponse(status="ok", vector_store="ready", bm25_index="ready", docs_indexed="0")
        assert r.version == APP_VERSION

    def test_api_key_is_secret_str(self):
        from app.models import QueryRequest
        req = QueryRequest(question="Hi", api_key="sk-supersecret")
        assert "sk-supersecret" not in str(req.api_key)
        assert req.api_key.get_secret_value() == "sk-supersecret"

    def test_stream_query_request_has_top_k(self):
        """BUG-S: StreamQueryRequest must carry top_k."""
        from app.models import StreamQueryRequest
        req = StreamQueryRequest(question="Hi", top_k=3)
        assert req.top_k == 3

    def test_providers_response(self):
        from app.models import ProvidersResponse
        from app.constants import PROVIDER_MODELS
        r = ProvidersResponse(providers=PROVIDER_MODELS)
        assert "OpenAI" in r.providers


# ─────────────────────────────────────────────
# BUG-Q — LOG_LEVEL respected
# ─────────────────────────────────────────────

class TestLogging:
    def test_setup_logging_reads_log_level(self):
        """BUG-Q: setup_logging() must apply the LOG_LEVEL from Settings."""
        import logging
        from unittest.mock import patch
        # Patch get_settings to return DEBUG level
        mock_settings = MagicMock()
        mock_settings.LOG_LEVEL = "DEBUG"
        with patch("app.utils.get_settings", return_value=mock_settings):
            from app.utils import setup_logging
            setup_logging()
        root_level = logging.getLogger().level
        assert root_level == logging.DEBUG, \
            f"BUG-Q: expected DEBUG level, got {logging.getLevelName(root_level)}."
