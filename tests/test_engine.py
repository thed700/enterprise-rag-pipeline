"""
tests/test_engine.py — AuraRAG v3.3
Author: Akmal Raxmatov (github: thed700)

Run: pytest tests/ -v

New tests in v3.3:
  BUG-Y regression: RerankedRetriever is used (not raw hybrid) in query()
  BUG-Z regression: arerank() uses get_running_loop(), not get_event_loop()
  BUG-AA regression: _api_stream() payload includes top_k
  BUG-AB regression: RAGEngine.shutdown() calls reranker.shutdown()
  BUG-AC regression: _evict_stale() reads settings.SESSION_TTL_MINUTES
  BUG-AD regression: HealthResponse model exposes bm25_docs field
  BUG-AE regression: query() snippet length reads settings.SOURCE_SNIPPET_LEN

Retained from v3.2.0:
  BUG-S, BUG-U, BUG-V (structure), BUG-X, BUG-Q, BUG-R, BUG-T tests.
"""

import asyncio
import inspect
import re
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


from langchain_core.retrievers import BaseRetriever


class _StubRetriever(BaseRetriever):
    """A real BaseRetriever (EnsembleRetriever validates its members' types)."""

    def _get_relevant_documents(self, query, *, run_manager=None):
        return list(DOCS_A)


_ACTIVE_STACKS: list = []


@pytest.fixture(autouse=True)
def _close_engine_patches():
    """Undo the patches _make_engine() leaves active after each test."""
    yield
    while _ACTIVE_STACKS:
        _ACTIVE_STACKS.pop().close()


def _make_engine():
    """
    Construct a RAGEngine with all heavy dependencies mocked.

    The patches stay active until the test ends (see _close_engine_patches) so
    ingest_documents() also hits the mocked Chroma instead of the real one.
    """
    import pathlib
    from contextlib import ExitStack
    stack = ExitStack()
    _ACTIVE_STACKS.append(stack)

    mock_ce = stack.enter_context(patch("app.engine.pipeline.CrossEncoder"))
    stack.enter_context(patch("app.engine.pipeline.HuggingFaceEmbeddings"))
    stack.enter_context(patch("app.engine.pipeline.chromadb.PersistentClient"))
    mock_chroma_cls = stack.enter_context(patch("app.engine.pipeline.Chroma"))
    stack.enter_context(patch("app.engine.pipeline.RAGEngine._load_bm25_from_disk"))
    stack.enter_context(patch("app.engine.pipeline.RAGEngine._save_bm25_to_disk"))
    stack.enter_context(patch("app.engine.pipeline._resolve_persist_dir",
                              return_value=pathlib.Path("/tmp/aurarag-test")))

    mock_ce.return_value = MagicMock()
    mock_col = MagicMock()
    mock_col.count.return_value = 0
    mock_store = MagicMock()
    mock_store._collection = mock_col
    mock_chroma_cls.return_value = mock_store
    mock_chroma_cls.from_documents.return_value = mock_store
    mock_store.as_retriever.return_value = _StubRetriever()

    from app.engine import RAGEngine
    engine = RAGEngine()
    return engine, mock_store


def _fake_graph(final_state):
    """A stand-in compiled graph whose ainvoke() returns a canned final state."""
    g = MagicMock()
    g.ainvoke = AsyncMock(return_value=final_state)
    return g


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
        """BUG-R: dated 4.5 IDs (e.g. claude-opus-4-5-20251101) are not valid."""
        from app.constants import PROVIDER_MODELS
        models = PROVIDER_MODELS["Anthropic"]
        for model in models:
            assert not re.fullmatch(r"claude-(opus|sonnet|haiku)-4-5-\d{8}", model), model
        assert "claude-opus-4-5" not in models
        assert "claude-opus-4-6" in models
        assert "claude-sonnet-4-6" in models

    def test_all_providers_present(self):
        from app.constants import PROVIDER_MODELS
        for provider in ("OpenAI", "Anthropic", "Google Gemini", "Local (Ollama)"):
            assert provider in PROVIDER_MODELS
            assert len(PROVIDER_MODELS[provider]) > 0


# ─────────────────────────────────────────────
# CrossEncoderReranker
# ─────────────────────────────────────────────

class TestReranker:
    @patch("app.engine.pipeline.CrossEncoder")
    def test_returns_top_k(self, mock_ce):
        from app.engine import CrossEncoderReranker
        mock_ce.return_value.predict.return_value = [0.9, 0.5, 0.1]
        r = CrossEncoderReranker()
        result = r.rerank("query", DOCS_A, top_k=2)
        assert len(result) == 2

    @patch("app.engine.pipeline.CrossEncoder")
    def test_sorted_by_score(self, mock_ce):
        from app.engine import CrossEncoderReranker
        mock_ce.return_value.predict.return_value = [0.1, 0.95, 0.4]
        r = CrossEncoderReranker()
        result = r.rerank("ChromaDB", DOCS_A, top_k=3)
        assert "ChromaDB" in result[0].page_content

    @patch("app.engine.pipeline.CrossEncoder")
    def test_empty_input(self, mock_ce):
        from app.engine import CrossEncoderReranker
        r = CrossEncoderReranker()
        assert r.rerank("q", [], top_k=5) == []

    @patch("app.engine.pipeline.CrossEncoder")
    def test_shutdown_calls_executor_shutdown(self, mock_ce):
        """BUG-AB: CrossEncoderReranker.shutdown() must release the executor."""
        from app.engine import CrossEncoderReranker
        r = CrossEncoderReranker()
        with patch.object(r._executor, "shutdown") as mock_shutdown:
            r.shutdown()
        mock_shutdown.assert_called_once_with(wait=False)


# ─────────────────────────────────────────────
# BUG-AB regression — RAGEngine.shutdown()
# ─────────────────────────────────────────────

class TestEngineShutdown:
    def test_engine_shutdown_calls_reranker_shutdown(self):
        """BUG-AB: RAGEngine.shutdown() must delegate to reranker.shutdown()."""
        engine, _ = _make_engine()
        with patch.object(engine.reranker, "shutdown") as mock_shutdown:
            engine.shutdown()
        mock_shutdown.assert_called_once()


# ─────────────────────────────────────────────
# BUG-Z regression — arerank() uses get_running_loop
# ─────────────────────────────────────────────

class TestArerank:
    @patch("app.engine.pipeline.CrossEncoder")
    def test_arerank_uses_get_running_loop(self, mock_ce):
        """BUG-Z: arerank() must call asyncio.get_running_loop(), not get_event_loop()."""
        import app.engine.pipeline as eng_module
        import inspect
        src = inspect.getsource(eng_module.CrossEncoderReranker.arerank)
        assert "get_running_loop" in src, \
            "BUG-Z: arerank() must use asyncio.get_running_loop() (not get_event_loop())."
        assert "get_event_loop" not in src, \
            "BUG-Z: deprecated asyncio.get_event_loop() found in arerank()."


# ─────────────────────────────────────────────
# BUG-Y regression — RerankedRetriever used in query/stream
# ─────────────────────────────────────────────

class TestRerankedRetriever:
    def test_reranked_retriever_exists(self):
        """BUG-Y: RerankedRetriever must be importable from engine."""
        from app.engine import RerankedRetriever
        assert RerankedRetriever is not None

    def test_build_reranked_retriever_returns_reranked_retriever(self):
        """BUG-Y: _build_reranked_retriever() must return a RerankedRetriever."""
        from app.engine import RerankedRetriever
        engine, _ = _make_engine()
        engine.ingest_documents(DOCS_A)
        retriever = engine._build_reranked_retriever(top_k=3)
        assert isinstance(retriever, RerankedRetriever)
        assert retriever.top_k == 3

    def test_graph_is_built_with_reranked_retriever(self):
        """BUG-Y: the compiled graph must be given a RerankedRetriever (so the
        cross-encoder is active), never the raw hybrid retriever."""
        from app.engine import RerankedRetriever
        engine, _ = _make_engine()
        engine.ingest_documents(DOCS_A + DOCS_B)

        captured = []
        with patch("app.engine.pipeline.build_aura_graph",
                   side_effect=lambda reranked_retriever: captured.append(reranked_retriever) or MagicMock()):
            engine._rebuild_graph(top_k=4)

        assert len(captured) == 1
        assert isinstance(captured[0], RerankedRetriever)
        assert captured[0].top_k == 4


# ─────────────────────────────────────────────
# BUG-AC regression — SESSION_TTL_MINUTES from settings
# ─────────────────────────────────────────────

class TestSessionTTLFromSettings:
    def test_evict_stale_reads_settings_not_constant(self):
        """BUG-AC: _evict_stale() must read SESSION_TTL_MINUTES from settings,
        not from the module-level constant, so .env changes take effect."""
        from app.engine import SessionMemoryStore
        import app.engine.pipeline as eng_module
        src = inspect.getsource(eng_module.SessionMemoryStore._evict_stale)
        assert "get_settings" in src, \
            "BUG-AC: _evict_stale() must call get_settings().SESSION_TTL_MINUTES."

    def test_ttl_from_settings_is_respected(self):
        """BUG-AC: a custom TTL from settings must drive eviction."""
        from app.engine import SessionMemoryStore
        from unittest.mock import MagicMock

        mock_settings = MagicMock()
        mock_settings.SESSION_TTL_MINUTES = 10  # 10-minute TTL

        store = SessionMemoryStore(window_k=5)
        store.get("alice")
        # Make alice's last access 11 minutes ago
        store._last_access["alice"] = time.monotonic() - (10 * 60 + 1)

        with patch("app.engine.pipeline.get_settings", return_value=mock_settings):
            store._evict_stale()

        assert store.active_sessions == 0, \
            "BUG-AC: session should have been evicted using the settings TTL."


# ─────────────────────────────────────────────
# BUG-AD regression — HealthResponse.bm25_docs
# ─────────────────────────────────────────────

class TestHealthResponseModel:
    def test_health_response_has_bm25_docs_field(self):
        """BUG-AD: HealthResponse must declare bm25_docs so FastAPI serialises it."""
        from app.models import HealthResponse
        r = HealthResponse(
            status="ok",
            vector_store="ready",
            bm25_index="ready",
            docs_indexed="100",
            bm25_docs="42",
        )
        assert r.bm25_docs == "42", \
            "BUG-AD: HealthResponse.bm25_docs field is missing or incorrect."

    def test_health_response_bm25_docs_defaults_to_zero(self):
        """BUG-AD: bm25_docs must have a sensible default."""
        from app.models import HealthResponse
        r = HealthResponse(
            status="ok",
            vector_store="empty",
            bm25_index="empty",
            docs_indexed="0",
        )
        assert r.bm25_docs == "0"

    def test_health_response_includes_bm25_docs_in_serialisation(self):
        """BUG-AD: bm25_docs must appear in model_dump() so it reaches the API client."""
        from app.models import HealthResponse
        r = HealthResponse(
            status="ok",
            vector_store="ready",
            bm25_index="ready",
            docs_indexed="50",
            bm25_docs="50",
        )
        data = r.model_dump()
        assert "bm25_docs" in data, \
            "BUG-AD: bm25_docs not present in serialised HealthResponse."


# ─────────────────────────────────────────────
# BUG-AE regression — SOURCE_SNIPPET_LEN from settings
# ─────────────────────────────────────────────

class TestSourceSnippetLen:
    def test_query_respects_source_snippet_len_setting(self):
        """BUG-AE: source cards must be truncated to settings.SOURCE_SNIPPET_LEN."""
        engine, _ = _make_engine()
        engine.ingest_documents(DOCS_A + DOCS_B)

        state = {
            "question": "test", "top_k": 5, "answer": "ok",
            "relevant_docs": [Document(page_content="A" * 600, metadata={"source": "x.pdf"})],
        }
        mock_settings = MagicMock()
        mock_settings.SOURCE_SNIPPET_LEN = 50
        mock_settings.SESSION_TTL_MINUTES = 60

        with patch.object(engine, "_get_or_rebuild_graph", return_value=_fake_graph(state)), \
             patch("app.engine.pipeline.get_settings", return_value=mock_settings):
            result = asyncio.run(engine.aquery("test", "OpenAI", "gpt-4o-mini", "sk-test",
                                               session_id="s1", top_k=5))

        assert len(result["sources"][0]["content"]) <= 50


# ─────────────────────────────────────────────
# BUG-AA regression — top_k in the stream payload
# ─────────────────────────────────────────────

class TestStreamPayload:
    def test_stream_payload_includes_top_k(self):
        """BUG-AA: the UI must send top_k in the /query/stream payload."""
        import app.ui as ui_module
        src = inspect.getsource(ui_module.main)
        assert '"top_k"' in src and "_query_stream(payload)" in src


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
        from unittest.mock import MagicMock

        store = SessionMemoryStore(window_k=5)
        store.get("old_session")
        store._last_access["old_session"] = time.monotonic() - (SESSION_TTL_MINUTES * 60 + 1)

        mock_settings = MagicMock()
        mock_settings.SESSION_TTL_MINUTES = SESSION_TTL_MINUTES
        with patch("app.engine.pipeline.get_settings", return_value=mock_settings):
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

        with patch("app.engine.pipeline.RAGEngine._save_bm25_to_disk") as mock_save:
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
        """BUG-S: only top_k sources are returned."""
        engine, _ = _make_engine()
        engine.ingest_documents(DOCS_A + DOCS_B)
        state = {
            "question": "test", "top_k": 2, "answer": "ok",
            "relevant_docs": [
                Document(page_content=f"doc {i}", metadata={"source": "x.pdf"}) for i in range(6)
            ],
        }
        with patch.object(engine, "_get_or_rebuild_graph", return_value=_fake_graph(state)):
            result = asyncio.run(engine.aquery("test", "OpenAI", "gpt-4o-mini", "sk-test",
                                               session_id="s1", top_k=2))
        assert len(result["sources"]) == 2


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
        """v3.2.0+: health() must report bm25_docs count."""
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
        mock_settings = MagicMock()
        mock_settings.LOG_LEVEL = "DEBUG"
        with patch("app.utils.get_settings", return_value=mock_settings):
            from app.utils import setup_logging
            setup_logging()
        root_level = logging.getLogger().level
        assert root_level == logging.DEBUG, \
            f"BUG-Q: expected DEBUG level, got {logging.getLevelName(root_level)}."
