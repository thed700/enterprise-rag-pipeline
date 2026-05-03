"""
engine.py - RAG Core Logic (LLM-Agnostic Edition)
Author: Akmal Raxmatov (github: thed700)

Enterprise-Grade RAG Pipeline with:
  - Hybrid Search (Dense + BM25)
  - Cross-Encoder Re-ranking
  - Dynamic LLM Factory (OpenAI / Anthropic / Google Gemini / Ollama)
"""

import logging
from typing import List, Tuple, Optional, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document, BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

from sentence_transformers import CrossEncoder
import chromadb

from app.utils import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────
# PROVIDER CONFIGURATION REGISTRY
# Maps provider names → available models
# ─────────────────────────────────────────────

PROVIDER_MODELS: Dict[str, List[str]] = {
    "OpenAI": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ],
    "Anthropic": [
        "claude-opus-4-5",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
    ],
    "Google Gemini": [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash",
        "gemini-2.5-pro",
    ],
    "Local (Ollama)": [
        "llama3",
        "llama3:8b",
        "llama3:70b",
        "mistral",
        "mixtral",
        "phi3",
        "gemma2",
    ],
}


# ─────────────────────────────────────────────
# SYSTEM PROMPT — Strict hallucination guard
# ─────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are a precise and trustworthy AI assistant for an enterprise knowledge base.
Your answers must be grounded EXCLUSIVELY in the provided context.

STRICT RULES:
1. If the answer is not explicitly found in the context, state: "I do not have enough information in the provided documents to answer this question."
2. Do NOT invent, extrapolate, or assume facts not present in the context.
3. Always cite the source document when possible.
4. Be concise, professional, and factual.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer (grounded strictly in the context above):"""

STRICT_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=SYSTEM_PROMPT_TEMPLATE,
)


# ─────────────────────────────────────────────
# LLM FACTORY — Dynamic provider instantiation
# ─────────────────────────────────────────────

def build_llm(
    provider: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    streaming: bool = False,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
) -> BaseLanguageModel:
    """
    Factory function: returns a LangChain-compatible chat model for the
    given provider/model/key triple.

    All credentials are passed at call-time — nothing is read from disk
    or environment variables here. This is the single point of truth for
    LLM construction.

    Args:
        provider:    One of PROVIDER_MODELS keys.
        model:       Model identifier string (provider-specific).
        api_key:     User-supplied API key (from st.session_state only).
        temperature: Sampling temperature (0 = deterministic).
        streaming:   Enable token-by-token streaming.
        callbacks:   LangChain callback handlers (e.g. StreamingStdOutCallbackHandler).

    Returns:
        Instantiated LangChain BaseChatModel.

    Raises:
        ValueError: Unknown provider name.
        ImportError: Required provider package not installed.
    """
    common_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "streaming": streaming,
        "callbacks": callbacks or [],
    }

    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            **common_kwargs,
        )

    elif provider == "Anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            anthropic_api_key=api_key,
            **common_kwargs,
        )

    elif provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            **common_kwargs,
        )

    elif provider == "Local (Ollama)":
        from langchain_community.chat_models import ChatOllama
        # Ollama runs locally — no API key required.
        return ChatOllama(
            model=model,
            **common_kwargs,
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose from: {list(PROVIDER_MODELS.keys())}"
        )


def validate_provider_config(
    provider: str,
    api_key: str,
) -> Tuple[bool, str]:
    """
    Lightweight check: does the provider + key combination look sane?
    Does NOT make a live API call — just validates format and presence.

    Returns:
        (is_valid: bool, message: str)
    """
    if provider == "Local (Ollama)":
        return True, "Ollama runs locally — no key needed."

    if not api_key or len(api_key.strip()) < 8:
        return False, "API key appears to be missing or too short."

    key = api_key.strip()

    prefix_map = {
        "OpenAI": "sk-",
        "Anthropic": "sk-ant-",
        "Google Gemini": "AI",
    }

    expected = prefix_map.get(provider, "")
    if expected and not key.startswith(expected):
        return (
            False,
            f"Key doesn't match expected {provider} format "
            f"(should start with '{expected}').",
        )

    return True, f"{provider} key looks valid ✓"


# ─────────────────────────────────────────────
# CROSS-ENCODER RE-RANKER
# ─────────────────────────────────────────────

class CrossEncoderReranker:
    """Cross-Encoder Re-ranker for improved retrieval precision."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading Cross-Encoder model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        """Re-rank documents using cross-encoder scores."""
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores: List[float] = self.model.predict(pairs).tolist()

        scored_docs: List[Tuple[float, Document]] = sorted(
            zip(scores, documents), key=lambda x: x[0], reverse=True
        )

        logger.info(
            f"Re-ranked {len(documents)} docs → returning top {top_k}. "
            f"Top score: {scored_docs[0][0]:.4f}"
        )
        return [doc for _, doc in scored_docs[:top_k]]


# ─────────────────────────────────────────────
# RAG ENGINE
# ─────────────────────────────────────────────

class RAGEngine:
    """
    Enterprise RAG Engine — LLM-Agnostic Edition.

    Pipeline:
      Ingest → Chunk → Embed → Store (ChromaDB)
      Query  → Hybrid Search (Dense + BM25) → Cross-Encoder Re-rank → LLM

    The LLM is NOT stored on the instance. It is built fresh per query
    call using the caller-supplied (provider, model, api_key) triple.
    This makes the engine fully stateless with respect to credentials.
    """

    def __init__(self) -> None:
        logger.info("Initialising RAGEngine (LLM-Agnostic)...")

        # Embedding model — shared, provider-independent
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
        )

        # ChromaDB persistent client
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR
        )
        self.vector_store: Optional[Chroma] = None

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        # Re-ranker
        self.reranker = CrossEncoderReranker()

        # Conversation memory (per-session; swap to Redis for multi-user)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        # Cached BM25 retriever (rebuilt on ingest)
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._all_docs: List[Document] = []

        logger.info("RAGEngine ready.")

    # ──────────────────────────────────────────
    # INGESTION LAYER
    # ──────────────────────────────────────────

    def ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Chunk → Embed → Store documents into ChromaDB + build BM25 index.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            Ingestion summary dict.
        """
        logger.info(f"Ingesting {len(documents)} raw document(s)...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks.")

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            client=self.chroma_client,
            collection_name=settings.CHROMA_COLLECTION,
        )

        # Build / refresh BM25 sparse index
        self._all_docs = chunks
        self._bm25_retriever = BM25Retriever.from_documents(chunks)
        self._bm25_retriever.k = 10

        summary = {"chunks_ingested": len(chunks), "status": "success"}
        logger.info(f"Ingestion complete: {summary}")
        return summary

    # ──────────────────────────────────────────
    # RETRIEVAL LAYER  (Hybrid Search preserved)
    # ──────────────────────────────────────────

    def _build_hybrid_retriever(self) -> EnsembleRetriever:
        """Combine dense (Chroma/MMR) + sparse (BM25) retrievers."""
        if self.vector_store is None or self._bm25_retriever is None:
            raise RuntimeError(
                "No documents ingested. Call ingest_documents() first."
            )

        dense_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 20},
        )

        return EnsembleRetriever(
            retrievers=[dense_retriever, self._bm25_retriever],
            weights=[0.6, 0.4],  # Dense retrieval weighted slightly higher
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Hybrid search + cross-encoder re-ranking (standalone, no LLM)."""
        hybrid = self._build_hybrid_retriever()
        candidates = hybrid.get_relevant_documents(query)
        logger.info(f"Hybrid retrieval returned {len(candidates)} candidates.")
        return self.reranker.rerank(query, candidates, top_k=top_k)

    # ──────────────────────────────────────────
    # GENERATION LAYER
    # ──────────────────────────────────────────

    def query(
        self,
        question: str,
        provider: str,
        model: str,
        api_key: str,
        streaming_callback: Optional[BaseCallbackHandler] = None,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline: retrieve → re-rank → generate.

        The LLM is built fresh each call from the supplied credentials.
        No keys are stored on the engine instance.

        Args:
            question:           Natural language question.
            provider:           LLM provider name (see PROVIDER_MODELS).
            model:              Model identifier string.
            api_key:            User-supplied API key (from session_state).
            streaming_callback: Optional LangChain streaming callback.

        Returns:
            Dict with 'answer', 'sources', and 'chat_history'.
        """
        if self.vector_store is None:
            return {
                "answer": "No documents have been ingested yet. Please upload and ingest files first.",
                "sources": [],
                "chat_history": [],
            }

        # Build LLM for this request — credentials never touch disk
        callbacks = [streaming_callback] if streaming_callback else []
        llm = build_llm(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=0.0,
            streaming=streaming_callback is not None,
            callbacks=callbacks,
        )

        hybrid_retriever = self._build_hybrid_retriever()

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=hybrid_retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": STRICT_PROMPT},
            return_source_documents=True,
            verbose=False,
        )

        logger.info(
            f"Running query via {provider}/{model}: '{question[:80]}...'"
        )
        result = chain({"question": question})

        # Re-rank source docs for transparency
        reranked_sources = self.reranker.rerank(
            question, result.get("source_documents", []), top_k=3
        )

        sources = [
            {
                "content": doc.page_content[:300],
                "metadata": doc.metadata,
            }
            for doc in reranked_sources
        ]

        logger.info("Query answered successfully.")
        return {
            "answer": result["answer"],
            "sources": sources,
            "chat_history": [
                m.content for m in self.memory.chat_memory.messages
            ],
        }

    # ──────────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────────

    def clear_memory(self) -> None:
        """Clear conversation history for a new session."""
        self.memory.clear()
        logger.info("Conversation memory cleared.")

    def health(self) -> Dict[str, str]:
        """Return engine health status."""
        return {
            "vector_store": "ready" if self.vector_store else "empty",
            "bm25_index": "ready" if self._bm25_retriever else "empty",
            "docs_indexed": str(len(self._all_docs)),
        }
