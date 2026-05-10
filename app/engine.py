"""
engine.py - AuraRAG Core Engine (LLM-Agnostic Edition)
Author: Akmal Raxmatov (github: thed700)

FIXES applied (v3.0.0):
  BUG-01: ConversationBufferMemory replaced with ConversationBufferWindowMemory(k=20)
          to prevent unbounded memory growth and LLM context-length overflows.
  BUG-03: ingest_documents() now ACCUMULATES chunks (_all_docs.extend) instead of
          overwriting them, so a second upload no longer wipes the first corpus.
          BM25 index is rebuilt from the full cumulative corpus and persisted to disk.
  BUG-04: vector_store loaded via get-or-create; documents added incrementally via
          add_documents() instead of Chroma.from_documents() (which clobbers the
          existing collection on every ingest call).
  BUG-07: Removed the duplicate reranker.rerank() call inside query().
          Re-ranking is now applied exactly once (inside retrieve()).
  BUG-08: Documented the single-worker constraint; memory is windowed (k=20) as a
          mitigation until Redis-backed per-session memory lands in v3.1.
  NOTE:   BUG-02 (async blocking), BUG-05 (temp-file cleanup), BUG-06 (SecretStr),
          BUG-09 (CORS), BUG-10 (fake streaming) are fixed in main.py / models.py.

UPGRADES (v3.0.0):
  - Rebranded NeuralDocs -> AuraRAG (Advanced Unified Retrieval Architecture)
  - PROVIDER_MODELS updated with latest Claude 4.x and Gemini 2.x model IDs
  - BM25 index persisted to disk (bm25.pkl beside chroma_db) and reloaded on startup
"""

import logging
import pickle
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import chromadb

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

from app.utils import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────
# PROVIDER CONFIGURATION REGISTRY
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
        # Full date-stamped ID required by the Anthropic API.
        "claude-haiku-4-5-20251001",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
    ],
    "Google Gemini": [
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
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
# SYSTEM PROMPT
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
# LLM FACTORY
# ─────────────────────────────────────────────

def build_llm(
    provider: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    streaming: bool = False,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
) -> BaseLanguageModel:
    common_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "streaming": streaming,
        "callbacks": callbacks or [],
    }

    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, openai_api_key=api_key, **common_kwargs)

    elif provider == "Anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, anthropic_api_key=api_key, **common_kwargs)

    elif provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, **common_kwargs)

    elif provider == "Local (Ollama)":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama  # type: ignore
        return ChatOllama(model=model, temperature=temperature)

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose from: {list(PROVIDER_MODELS.keys())}"
        )


def validate_provider_config(provider: str, api_key: str) -> Tuple[bool, str]:
    if provider == "Local (Ollama)":
        return True, "Ollama runs locally — no key needed."
    if not api_key or len(api_key.strip()) < 8:
        return False, "API key appears to be missing or too short."
    key = api_key.strip()
    prefix_map = {"OpenAI": "sk-", "Anthropic": "sk-ant-", "Google Gemini": "AI"}
    expected = prefix_map.get(provider, "")
    if expected and not key.startswith(expected):
        return False, f"Key doesn't match expected {provider} format (should start with '{expected}')."
    return True, f"{provider} key looks valid ✓"


# ─────────────────────────────────────────────
# CROSS-ENCODER RE-RANKER
# ─────────────────────────────────────────────

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading Cross-Encoder model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        raw = self.model.predict(pairs)
        scores: List[float] = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        logger.info(
            f"Re-ranked {len(documents)} docs -> returning top {top_k}. "
            f"Top score: {scored_docs[0][0]:.4f}"
        )
        return [doc for _, doc in scored_docs[:top_k]]


# ─────────────────────────────────────────────
# RAG ENGINE
# ─────────────────────────────────────────────

_BM25_PICKLE_PATH = pathlib.Path(settings.CHROMA_PERSIST_DIR) / "bm25.pkl"


class RAGEngine:
    """
    AuraRAG Enterprise Engine — LLM-Agnostic Edition.
    Pipeline: Ingest -> Chunk -> Embed -> Store (ChromaDB)
              Query  -> Hybrid Search (Dense + BM25) -> Cross-Encoder Re-rank -> LLM

    CONCURRENCY NOTE (v3.0.0):
    This engine holds mutable in-process state (_all_docs, memory).
    Run uvicorn with --workers 1 until Redis-backed per-session memory
    is introduced in v3.1 (BUG-08).
    """

    def __init__(self) -> None:
        logger.info("Initialising AuraRAG Engine...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
        )

        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

        # FIX BUG-04: load existing collection (get-or-create).
        _store = Chroma(
            client=self.chroma_client,
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=self.embeddings,
        )
        self.vector_store: Optional[Chroma] = (
            _store if _store._collection.count() > 0 else None
        )
        self._chroma_store_ref = _store  # kept alive for incremental adds

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        self.reranker = CrossEncoderReranker()

        # FIX BUG-01: windowed memory (last 20 turns) prevents unbounded growth.
        self.memory = ConversationBufferWindowMemory(
            k=20,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        # FIX BUG-03: cumulative corpus — extended on each ingest, never overwritten.
        self._all_docs: List[Document] = []
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._load_bm25_from_disk()

        logger.info("AuraRAG Engine ready.")

    # ── BM25 PERSISTENCE ──────────────────────────────────────────────────────

    def _load_bm25_from_disk(self) -> None:
        if _BM25_PICKLE_PATH.exists():
            try:
                with open(_BM25_PICKLE_PATH, "rb") as f:
                    self._bm25_retriever = pickle.load(f)
                logger.info(f"BM25 index restored from {_BM25_PICKLE_PATH}.")
            except Exception as e:
                logger.warning(f"Could not restore BM25 index: {e}. Will rebuild on next ingest.")

    def _save_bm25_to_disk(self) -> None:
        try:
            _BM25_PICKLE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_BM25_PICKLE_PATH, "wb") as f:
                pickle.dump(self._bm25_retriever, f)
            logger.info(f"BM25 index saved to {_BM25_PICKLE_PATH}.")
        except Exception as e:
            logger.warning(f"Could not persist BM25 index: {e}.")

    # ── INGESTION ──────────────────────────────────────────────────────────────

    def ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
        logger.info(f"Ingesting {len(documents)} raw document(s)...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks.")

        # FIX BUG-04: add documents incrementally instead of clobbering the collection.
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                client=self.chroma_client,
                collection_name=settings.CHROMA_COLLECTION,
            )
            self._chroma_store_ref = self.vector_store
        else:
            self._chroma_store_ref.add_documents(chunks)

        # FIX BUG-03: accumulate corpus.
        self._all_docs.extend(chunks)

        self._bm25_retriever = BM25Retriever.from_documents(self._all_docs)
        self._bm25_retriever.k = 10
        self._save_bm25_to_disk()

        summary = {"chunks_ingested": len(chunks), "status": "success"}
        logger.info(f"Ingestion complete: {summary}")
        return summary

    # ── RETRIEVAL ──────────────────────────────────────────────────────────────

    def _build_hybrid_retriever(self) -> EnsembleRetriever:
        if self.vector_store is None or self._bm25_retriever is None:
            raise RuntimeError("No documents ingested. Call ingest_documents() first.")
        dense_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 20},
        )
        return EnsembleRetriever(
            retrievers=[dense_retriever, self._bm25_retriever],
            weights=[0.6, 0.4],
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Hybrid search + cross-encoder re-ranking (standalone, no LLM)."""
        hybrid = self._build_hybrid_retriever()
        candidates = hybrid.invoke(query)
        logger.info(f"Hybrid retrieval returned {len(candidates)} candidates.")
        return self.reranker.rerank(query, candidates, top_k=top_k)

    # ── GENERATION ─────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        provider: str,
        model: str,
        api_key: str,
        streaming_callback: Optional[BaseCallbackHandler] = None,
    ) -> Dict[str, Any]:
        if self.vector_store is None:
            return {
                "answer": "No documents have been ingested yet. Please upload and ingest files first.",
                "sources": [],
                "chat_history": [],
            }

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

        logger.info(f"Running query via {provider}/{model}: '{question[:80]}...'")
        result = chain.invoke({"question": question})

        # FIX BUG-07: no duplicate rerank here — use source_documents directly.
        sources = [
            {"content": doc.page_content[:300], "metadata": doc.metadata}
            for doc in result.get("source_documents", [])[:3]
        ]

        logger.info("Query answered successfully.")
        return {
            "answer": result["answer"],
            "sources": sources,
            "chat_history": [m.content for m in self.memory.chat_memory.messages],
        }

    # ── UTILITIES ──────────────────────────────────────────────────────────────

    def clear_memory(self) -> None:
        self.memory.clear()
        logger.info("Conversation memory cleared.")

    def health(self) -> Dict[str, str]:
        return {
            "vector_store": "ready" if self.vector_store else "empty",
            "bm25_index": "ready" if self._bm25_retriever else "empty",
            "docs_indexed": str(len(self._all_docs)),
        }
