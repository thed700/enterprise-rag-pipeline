"""
tests/test_engine.py
Unit tests for the Enterprise RAG Pipeline.
Author: Akmal Raxmatov (github: thed700)

Run: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import Document

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


# ─────────────────────────────────────────────
# Cross-Encoder Re-ranker Tests
# ─────────────────────────────────────────────

class TestCrossEncoderReranker:
    @patch("app.engine.CrossEncoder")
    def test_rerank_returns_top_k(self, mock_cross_encoder_cls):
        """Re-ranker must return at most top_k documents."""
        from app.engine import CrossEncoderReranker

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.5, 0.1]
        mock_cross_encoder_cls.return_value = mock_model

        reranker = CrossEncoderReranker()
        result = reranker.rerank("What is LangChain?", SAMPLE_DOCS, top_k=2)

        assert len(result) == 2

    @patch("app.engine.CrossEncoder")
    def test_rerank_orders_by_score(self, mock_cross_encoder_cls):
        """Highest-scored document should be first."""
        from app.engine import CrossEncoderReranker

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.95, 0.4]
        mock_cross_encoder_cls.return_value = mock_model

        reranker = CrossEncoderReranker()
        result = reranker.rerank("ChromaDB", SAMPLE_DOCS, top_k=3)

        # Second doc has highest score (0.95)
        assert "ChromaDB" in result[0].page_content

    @patch("app.engine.CrossEncoder")
    def test_rerank_empty_input(self, mock_cross_encoder_cls):
        """Re-ranker should handle empty doc list gracefully."""
        from app.engine import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", [], top_k=5)
        assert result == []


# ─────────────────────────────────────────────
# RAGEngine — Ingestion Tests
# ─────────────────────────────────────────────

class TestRAGEngineIngestion:
    @patch("app.engine.ChatOpenAI")
    @patch("app.engine.chromadb.PersistentClient")
    @patch("app.engine.HuggingFaceEmbeddings")
    @patch("app.engine.CrossEncoder")
    @patch("app.engine.Chroma.from_documents")
    def test_ingest_returns_chunk_count(
        self,
        mock_chroma_from_docs,
        mock_cross_encoder,
        mock_embeddings,
        mock_chroma_client,
        mock_llm,
    ):
        """Ingestion should return a positive chunk count."""
        from app.engine import RAGEngine

        mock_cross_encoder.return_value = MagicMock()
        mock_chroma_from_docs.return_value = MagicMock()

        engine = RAGEngine()
        result = engine.ingest_documents(SAMPLE_DOCS)

        assert result["status"] == "success"
        assert result["chunks_ingested"] > 0

    @patch("app.engine.ChatOpenAI")
    @patch("app.engine.chromadb.PersistentClient")
    @patch("app.engine.HuggingFaceEmbeddings")
    @patch("app.engine.CrossEncoder")
    @patch("app.engine.Chroma.from_documents")
    def test_ingest_builds_bm25_index(
        self,
        mock_chroma_from_docs,
        mock_cross_encoder,
        mock_embeddings,
        mock_chroma_client,
        mock_llm,
    ):
        """After ingestion, BM25 retriever must be initialised."""
        from app.engine import RAGEngine

        mock_cross_encoder.return_value = MagicMock()
        mock_chroma_from_docs.return_value = MagicMock()

        engine = RAGEngine()
        engine.ingest_documents(SAMPLE_DOCS)

        assert engine._bm25_retriever is not None


# ─────────────────────────────────────────────
# RAGEngine — Health Tests
# ─────────────────────────────────────────────

class TestRAGEngineHealth:
    @patch("app.engine.ChatOpenAI")
    @patch("app.engine.chromadb.PersistentClient")
    @patch("app.engine.HuggingFaceEmbeddings")
    @patch("app.engine.CrossEncoder")
    def test_health_before_ingest(
        self, mock_cross_encoder, mock_embeddings, mock_chroma_client, mock_llm
    ):
        """Before ingestion, health should report 'empty'."""
        from app.engine import RAGEngine

        mock_cross_encoder.return_value = MagicMock()
        engine = RAGEngine()
        h = engine.health()

        assert h["vector_store"] == "empty"
        assert h["bm25_index"] == "empty"

    @patch("app.engine.ChatOpenAI")
    @patch("app.engine.chromadb.PersistentClient")
    @patch("app.engine.HuggingFaceEmbeddings")
    @patch("app.engine.CrossEncoder")
    @patch("app.engine.Chroma.from_documents")
    def test_health_after_ingest(
        self,
        mock_chroma_from_docs,
        mock_cross_encoder,
        mock_embeddings,
        mock_chroma_client,
        mock_llm,
    ):
        """After ingestion, health should report 'ready'."""
        from app.engine import RAGEngine

        mock_cross_encoder.return_value = MagicMock()
        mock_chroma_from_docs.return_value = MagicMock()

        engine = RAGEngine()
        engine.ingest_documents(SAMPLE_DOCS)
        h = engine.health()

        assert h["vector_store"] == "ready"
        assert h["bm25_index"] == "ready"


# ─────────────────────────────────────────────
# Pydantic Schema Tests
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
