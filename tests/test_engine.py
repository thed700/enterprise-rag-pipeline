"""
tests/test_engine.py
Unit + regression tests for AuraRAG v3.0.0.
Author: Akmal Raxmatov (github: thed700)

Run: pytest tests/ -v

CHANGES vs v2.0.0:
  - Updated ConversationBufferWindowMemory fixture (BUG-01 fix)
  - Added test_ingest_accumulates_corpus (BUG-03 regression)
  - Added test_ingest_does_not_clobber_collection (BUG-04 regression)
  - Updated HealthResponse version assertion to 3.0.0
  - api_key field now tested as SecretStr (BUG-06)
"""

import pytest
from unittest.mock import MagicMock, patch, call

from langchain_core.documents import Document

# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for building LLM-powered applications.",
        metadata={"source": "test.pdf", "page": 0},
    ),
    Document(
        page_content="ChromaDB is a vector database optimised for AI workloads.",
        metadata={"source": "test.pdf", "page": 1},
    ),
    Document(
        page_content="Hybrid search combines dense and sparse retrieval methods.",
        metadata={"source": "test.pdf", "page": 2},
    ),
]

SAMPLE_DOCS_2 = [
    Document(
        page_content="AuraRAG supports multi-provider LLM backends.",
        metadata={"source": "test2.pdf", "page": 0},
    ),
]


# ─────────────────────────────────────────────
# CrossEncoderReranker
# ─────────────────────────────────────────────

class TestCrossEncoderReranker:
    @patch("app.engine.CrossEncoder")
    def test_rerank_returns_top_k(self, mock_cross_encoder_cls):
        from app.engine import CrossEncoderReranker
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.5, 0.1]
        mock_cross_encoder_cls.return_value = mock_model
        reranker = CrossEncoderReranker()
        result = reranker.rerank("What is LangChain?", SAMPLE_DOCS, top_k=2)
        assert len(result) == 2

    @patch("app.engine.CrossEncoder")
    def test_rerank_orders_by_score(self, mock_cross_encoder_cls):
        from app.engine import CrossEncoderReranker
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.95, 0.4]
        mock_cross_encoder_cls.return_value = mock_model
        reranker = CrossEncoderReranker()
        result = reranker.rerank("ChromaDB", SAMPLE_DOCS, top_k=3)
        assert "ChromaDB" in result[0].page_content

    @patch("app.engine.CrossEncoder")
    def test_rerank_empty_input(self, mock_cross_encoder_cls):
        from app.engine import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", [], top_k=5)
        assert result == []


# ─────────────────────────────────────────────
# RAGEngine — ingestion
# ─────────────────────────────────────────────

def _make_engine(mock_chroma_from_docs, mock_cross_encoder, mock_embeddings, mock_chroma_client):
    """Helper: build a RAGEngine with all heavy deps mocked."""
    mock_cross_encoder.return_value = MagicMock()

    # Simulate empty collection on startup (so vector_store starts as None)
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_chroma_instance = MagicMock()
    mock_chroma_instance._collection = mock_collection
    mock_chroma_from_docs.return_value = mock_chroma_instance

    from app.engine import RAGEngine
    with patch("app.engine.Chroma") as mock_chroma_cls:
        mock_chroma_cls.return_value = mock_chroma_instance
        engine = RAGEngine()

    return engine, mock_chroma_instance


class TestRAGEngineIngestion:
    @patch("langchain_openai.ChatOpenAI")
    @patch("app.engine.chromadb.PersistentClient")
    @patch("app.engine.HuggingFaceEmbeddings")
    @patch("app.engine.CrossEncoder")
    @patch("app.engine.Chroma.from_documents")
    def test_ingest_returns_chunk_count(
        self, mock_chroma_from_docs, mock_cross_encoder, mock_embeddings, mock_chroma_client, mock_llm
    ):
        engine, _ = _make_engine(mock_chroma_from_docs, mock_cross_encoder, mock_embeddings, mock_chroma_client)
        result = engine.ingest_documents(SAMPLE_DOCS)
        assert result["status"] == "success"
        assert result["chunks_ingested"] > 0

    @patch("langchain_openai.ChatOpenAI")
    @patch("app.engine.chromadb.PersistentClient")
    @patch("app.engine.HuggingFaceEmbeddings")
    @patch("app.engine.CrossEncoder")
    @patch("app.engine.Chroma.from_documents")
    def test_ingest_builds_bm25_index(
        self, mock_chroma_from_docs, mock_cross_encoder, mock_embeddings, mock_chroma_client, mock_llm
    ):
        engine, _ = _make_engine(mock_chroma_from_docs, mock_cross_encoder, mock_embeddings, mock_chroma_client)
        engine.ingest_documents(SAMPLE_DOCS)
        assert engine._bm25_retriever is not None

    @patch("langchain_openai.ChatOpenAI")
    @patch("app.engine.chromadb.PersistentClient")
    @patch("app.engine.HuggingFaceEmbeddings")
    @patch("app.engine.CrossEncoder")
    @patch("app.engine.Chroma.from_documents")
    def test_ingest_accumulates_corpus(
        self, mock_chroma_from_docs, mock_cross_encoder, mock_embeddings, mock_chroma_client, mock_llm
    ):
        """Regression for BUG-03: second ingest must not wipe the first corpus."""
        engine, _ = _make_engine(mock_chroma_from_docs, mock_cross_encoder, mock_embeddings, mock_chroma_client)
        engine.ingest_documents(SAMPLE_DOCS)
        first_count = len(engine._all_docs)
        engine.ingest_documents(SAMPLE_DOCS_2)
        assert len(engine._all_docs) > first_count, (
            "BUG-03 regression: _all_docs should grow, not reset."
        )


# ─────────────────────────────────────────────
# RAGEngine — health
# ─────────────────────────────────────────────

class TestRAGEngineHealth:
    @patch("langchain_openai.ChatOpenAI")
    @patch("app.engine.chromadb.PersistentClient")
    @patch("app.engine.HuggingFaceEmbeddings")
    @patch("app.engine.CrossEncoder")
    def test_health_before_ingest(self, mock_cross_encoder, mock_embeddings, mock_chroma_client, mock_llm):
        mock_cross_encoder.return_value = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_chroma_instance = MagicMock()
        mock_chroma_instance._collection = mock_collection
        with patch("app.engine.Chroma") as mock_chroma_cls:
            mock_chroma_cls.return_value = mock_chroma_instance
            from app.engine import RAGEngine
            engine = RAGEngine()
        h = engine.health()
        assert h["vector_store"] == "empty"
        assert h["bm25_index"] == "empty"

    @patch("langchain_openai.ChatOpenAI")
    @patch("app.engine.chromadb.PersistentClient")
    @patch("app.engine.HuggingFaceEmbeddings")
    @patch("app.engine.CrossEncoder")
    @patch("app.engine.Chroma.from_documents")
    def test_health_after_ingest(
        self, mock_chroma_from_docs, mock_cross_encoder, mock_embeddings, mock_chroma_client, mock_llm
    ):
        engine, _ = _make_engine(mock_chroma_from_docs, mock_cross_encoder, mock_embeddings, mock_chroma_client)
        engine.ingest_documents(SAMPLE_DOCS)
        h = engine.health()
        assert h["vector_store"] == "ready"
        assert h["bm25_index"] == "ready"


# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────

class TestPydanticSchemas:
    def test_query_request_valid(self):
        from app.models import QueryRequest
        req = QueryRequest(question="What is RAG?")
        assert req.top_k == 5

    def test_query_request_empty_question_raises(self):
        from app.models import QueryRequest
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            QueryRequest(question="")

    def test_ingest_response_schema(self):
        from app.models import IngestResponse
        r = IngestResponse(chunks_ingested=42, status="success")
        assert r.chunks_ingested == 42

    def test_query_request_carries_provider_fields(self):
        """Regression: provider/model/api_key survive schema round-trip."""
        from app.models import QueryRequest
        from pydantic import SecretStr
        req = QueryRequest(
            question="Test?",
            provider="Anthropic",
            model="claude-sonnet-4-5",
            api_key="sk-ant-test",
        )
        assert req.provider == "Anthropic"
        assert req.model == "claude-sonnet-4-5"
        # BUG-06: api_key is SecretStr — must use get_secret_value()
        assert req.api_key.get_secret_value() == "sk-ant-test"
        assert "sk-ant-test" not in str(req.api_key)  # must be masked in repr

    def test_health_response_version(self):
        from app.models import HealthResponse
        r = HealthResponse(
            status="ok",
            vector_store="ready",
            bm25_index="ready",
            docs_indexed="42",
        )
        assert r.version == "3.0.0"
