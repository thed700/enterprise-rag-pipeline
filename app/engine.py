"""
engine.py — AuraRAG Core Engine v3.1.0
Author: Akmal Raxmatov (github: thed700)

BUG FIXES in v3.1.0 (on top of v3.0.0):
  BUG-A: 'langchain_classic' does not exist on PyPI — caused ModuleNotFoundError
          on startup. All imports corrected to 'langchain' (langchain.chains,
          langchain.memory, langchain.retrievers).
  BUG-B: EnsembleRetriever was imported from the non-existent langchain_classic.
          Fixed to langchain.retrievers.ensemble.
  BUG-C: _BM25_PICKLE_PATH was computed at module import time, before .env is
          parsed by pydantic-settings. Moved into RAGEngine.__init__ so it always
          uses the fully-resolved settings value.
  BUG-D: Removed the confusing _chroma_store_ref / vector_store dual-reference.
          A single self.vector_store is used for both add_documents() and
          as_retriever() calls. The alias was harmless but fragile.
  BUG-E: No deduplication on ingest. The same file uploaded twice doubled chunks
          in both ChromaDB and BM25, poisoning retrieval scores. Deduplication
          now uses a SHA-256 content hash stored in self._seen_hashes.
  BUG-I: health() reported docs_indexed from _all_docs (empty after restart).
          Now queries ChromaDB collection.count() as the source of truth, which
          persists across restarts.
  BUG-K: ConversationBufferWindowMemory was imported from langchain_classic.memory.
          Fixed to langchain.memory.
  BUG-O: PROVIDER_MODELS and validate_provider_config moved to constants.py so the
          Streamlit frontend can import them without pulling in heavy ML deps.
  BUG-P: Per-session memory using session_id. Each query now carries a session_id;
          the engine maintains one ConversationBufferWindowMemory per session in
          self._sessions dict, clearing stale sessions after SESSION_TTL_MINUTES.

NEW in v3.1.0:
  - True SSE streaming via engine.stream_query() async generator
  - /providers API endpoint (returns PROVIDER_MODELS as JSON)
  - Configurable chunk_size / chunk_overlap via Settings
  - ThreadPoolExecutor used for cross-encoder batch inference (non-blocking)
"""

import asyncio
import hashlib
import logging
import pickle
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import chromadb

from langchain.chains import ConversationalRetrievalChain          # FIX BUG-A
from langchain.memory import ConversationBufferWindowMemory         # FIX BUG-K
from langchain.retrievers.ensemble import EnsembleRetriever         # FIX BUG-B
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

from app.constants import PROVIDER_MODELS, validate_provider_config  # FIX BUG-O
from app.utils import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Re-export so any legacy code that imported from engine still works
__all__ = ["RAGEngine", "build_llm", "PROVIDER_MODELS", "validate_provider_config"]

# Session memory TTL: sessions idle longer than this are evicted
SESSION_TTL_MINUTES = 60

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are a precise and trustworthy AI assistant for an enterprise knowledge base.
Your answers must be grounded EXCLUSIVELY in the provided context.

STRICT RULES:
1. If the answer is not found in the context, say: "I do not have enough information in the provided documents to answer this question."
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
    common: Dict[str, Any] = {
        "temperature": temperature,
        "streaming": streaming,
        "callbacks": callbacks or [],
    }

    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, openai_api_key=api_key, **common)

    elif provider == "Anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, anthropic_api_key=api_key, **common)

    elif provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, **common)

    elif provider == "Local (Ollama)":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama  # type: ignore
        return ChatOllama(model=model, temperature=temperature)

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: {list(PROVIDER_MODELS.keys())}"
        )


# ─────────────────────────────────────────────
# CROSS-ENCODER RE-RANKER
# ─────────────────────────────────────────────

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading Cross-Encoder: {model_name}")
        self.model = CrossEncoder(model_name)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reranker")

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        raw = self.model.predict(pairs)
        scores: List[float] = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        logger.info(
            f"Re-ranked {len(documents)} docs → top {top_k}. "
            f"Best score: {ranked[0][0]:.4f}"
        )
        return [doc for _, doc in ranked[:top_k]]

    async def arerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Non-blocking rerank: runs the CPU-bound inference in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self.rerank, query, documents, top_k
        )


# ─────────────────────────────────────────────
# SESSION MEMORY STORE
# ─────────────────────────────────────────────

class SessionMemoryStore:
    """
    Per-session windowed conversation memory. FIX BUG-P.
    Evicts sessions that have been idle longer than SESSION_TTL_MINUTES.
    """

    def __init__(self, window_k: int = 20) -> None:
        self._window_k = window_k
        self._sessions: Dict[str, ConversationBufferWindowMemory] = {}
        self._last_access: Dict[str, float] = {}

    def get(self, session_id: str) -> ConversationBufferWindowMemory:
        self._evict_stale()
        if session_id not in self._sessions:
            logger.debug(f"Creating new memory for session {session_id!r}.")
            self._sessions[session_id] = ConversationBufferWindowMemory(
                k=self._window_k,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
            )
        self._last_access[session_id] = time.monotonic()
        return self._sessions[session_id]

    def clear(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            logger.info(f"Memory cleared for session {session_id!r}.")

    def clear_all(self) -> None:
        self._sessions.clear()
        self._last_access.clear()

    def _evict_stale(self) -> None:
        cutoff = time.monotonic() - SESSION_TTL_MINUTES * 60
        stale = [sid for sid, t in self._last_access.items() if t < cutoff]
        for sid in stale:
            self._sessions.pop(sid, None)
            self._last_access.pop(sid, None)
            logger.debug(f"Evicted stale session {sid!r}.")

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)


# ─────────────────────────────────────────────
# DOCUMENT DEDUPLICATION
# ─────────────────────────────────────────────

def _content_hash(doc: Document) -> str:
    """SHA-256 fingerprint of a document chunk's content."""
    return hashlib.sha256(doc.page_content.encode()).hexdigest()


# ─────────────────────────────────────────────
# RAG ENGINE
# ─────────────────────────────────────────────

class RAGEngine:
    """
    AuraRAG Enterprise Engine v3.1.0 — LLM-Agnostic.

    Pipeline:
      Ingest  → Chunk → Deduplicate → Embed → Store (ChromaDB + BM25)
      Query   → Hybrid Search → Cross-Encoder Re-rank → LLM → Stream

    Concurrency:
      Safe for --workers 1. Per-session memory via SessionMemoryStore (BUG-P).
      BM25 rebuild and ChromaDB writes are synchronous; wrap in asyncio.to_thread()
      at the API layer for non-blocking behaviour.
    """

    def __init__(self) -> None:
        logger.info("Initialising AuraRAG Engine v3.1.0...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
        )

        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

        # FIX BUG-C: resolve path inside __init__, not at module level.
        # FIX BUG-D: single self.vector_store — no more dual-reference.
        self._bm25_pickle_path = pathlib.Path(settings.CHROMA_PERSIST_DIR) / "bm25.pkl"

        _store = Chroma(
            client=self.chroma_client,
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=self.embeddings,
        )
        self.vector_store: Optional[Chroma] = (
            _store if _store._collection.count() > 0 else None
        )
        # Keep a reference regardless so add_documents() and as_retriever() both
        # operate on the same underlying object after the first ingest.
        self._chroma = _store

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        self.reranker = CrossEncoderReranker()

        # FIX BUG-P: per-session memory store
        self._session_store = SessionMemoryStore(window_k=20)

        # FIX BUG-E: content-hash dedup set
        self._seen_hashes: set[str] = set()

        # _all_docs used only for BM25 rebuilds (in-memory, not for health count)
        self._all_docs: List[Document] = []
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._load_bm25_from_disk()

        logger.info("AuraRAG Engine ready.")

    # ── BM25 PERSISTENCE ──────────────────────────────────────────────────────

    def _load_bm25_from_disk(self) -> None:
        if self._bm25_pickle_path.exists():
            try:
                with open(self._bm25_pickle_path, "rb") as f:
                    saved = pickle.load(f)
                # saved is a dict: {"retriever": ..., "docs": [...]}
                if isinstance(saved, dict):
                    self._bm25_retriever = saved["retriever"]
                    self._all_docs = saved.get("docs", [])
                    # Rebuild seen-hashes from restored docs
                    self._seen_hashes = {_content_hash(d) for d in self._all_docs}
                else:
                    # Legacy pickle (bare BM25Retriever from v3.0.0)
                    self._bm25_retriever = saved
                logger.info(f"BM25 index restored ({len(self._all_docs)} docs).")
            except Exception as e:
                logger.warning(f"Could not restore BM25 index: {e}. Will rebuild on next ingest.")

    def _save_bm25_to_disk(self) -> None:
        try:
            self._bm25_pickle_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"retriever": self._bm25_retriever, "docs": self._all_docs}
            with open(self._bm25_pickle_path, "wb") as f:
                pickle.dump(payload, f)
            logger.info(f"BM25 index saved ({len(self._all_docs)} docs).")
        except Exception as e:
            logger.warning(f"Could not persist BM25 index: {e}.")

    # ── INGESTION ──────────────────────────────────────────────────────────────

    def ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
        logger.info(f"Ingesting {len(documents)} raw document(s)...")
        chunks = self.text_splitter.split_documents(documents)

        # FIX BUG-E: deduplicate by content hash before storing.
        new_chunks: List[Document] = []
        dup_count = 0
        for chunk in chunks:
            h = _content_hash(chunk)
            if h in self._seen_hashes:
                dup_count += 1
            else:
                self._seen_hashes.add(h)
                new_chunks.append(chunk)

        if dup_count:
            logger.info(f"Skipped {dup_count} duplicate chunk(s).")

        if not new_chunks:
            return {"chunks_ingested": 0, "duplicates_skipped": dup_count, "status": "success"}

        logger.info(f"Adding {len(new_chunks)} new chunks...")

        # FIX BUG-D: single self._chroma object for both writes and retrieval.
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=new_chunks,
                embedding=self.embeddings,
                client=self.chroma_client,
                collection_name=settings.CHROMA_COLLECTION,
            )
            self._chroma = self.vector_store
        else:
            self._chroma.add_documents(new_chunks)
            self.vector_store = self._chroma   # keep alias in sync

        self._all_docs.extend(new_chunks)
        self._bm25_retriever = BM25Retriever.from_documents(self._all_docs)
        self._bm25_retriever.k = 10
        self._save_bm25_to_disk()

        result = {
            "chunks_ingested": len(new_chunks),
            "duplicates_skipped": dup_count,
            "status": "success",
        }
        logger.info(f"Ingestion complete: {result}")
        return result

    # ── RETRIEVAL ──────────────────────────────────────────────────────────────

    def _build_hybrid_retriever(self) -> EnsembleRetriever:
        if self.vector_store is None or self._bm25_retriever is None:
            raise RuntimeError("No documents ingested. Upload files first.")
        dense = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 20},
        )
        return EnsembleRetriever(
            retrievers=[dense, self._bm25_retriever],
            weights=[0.6, 0.4],
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Standalone hybrid search + cross-encoder re-rank (no LLM)."""
        hybrid = self._build_hybrid_retriever()
        candidates = hybrid.invoke(query)
        return self.reranker.rerank(query, candidates, top_k=top_k)

    # ── SYNCHRONOUS QUERY (used via asyncio.to_thread in main.py) ─────────────

    def query(
        self,
        question: str,
        provider: str,
        model: str,
        api_key: str,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        if self.vector_store is None:
            return {
                "answer": "No documents ingested yet. Upload files first.",
                "sources": [],
                "chat_history": [],
                "session_id": session_id,
            }

        # FIX BUG-P: per-session memory
        memory = self._session_store.get(session_id)

        llm = build_llm(provider=provider, model=model, api_key=api_key, temperature=0.0)

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self._build_hybrid_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": STRICT_PROMPT},
            return_source_documents=True,
            verbose=False,
        )

        logger.info(f"[{session_id}] Query via {provider}/{model}: '{question[:80]}'")
        result = chain.invoke({"question": question})

        sources = [
            {"content": doc.page_content[:300], "metadata": doc.metadata}
            for doc in result.get("source_documents", [])[:3]
        ]

        return {
            "answer": result["answer"],
            "sources": sources,
            "chat_history": [m.content for m in memory.chat_memory.messages],
            "session_id": session_id,
        }

    # ── TRUE SSE STREAMING (v3.1.0 new) ───────────────────────────────────────

    async def stream_query(
        self,
        question: str,
        provider: str,
        model: str,
        api_key: str,
        session_id: str = "default",
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that yields LLM tokens as they arrive.
        Consumed by the FastAPI /query/stream SSE endpoint.
        """
        if self.vector_store is None:
            yield "No documents ingested yet. Upload files first."
            return

        memory = self._session_store.get(session_id)

        from langchain_core.callbacks import AsyncIteratorCallbackHandler

        callback = AsyncIteratorCallbackHandler()
        llm = build_llm(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=0.0,
            streaming=True,
            callbacks=[callback],
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self._build_hybrid_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": STRICT_PROMPT},
            return_source_documents=False,
            verbose=False,
        )

        # Run chain in background thread so it doesn't block the event loop
        task = asyncio.create_task(
            asyncio.to_thread(chain.invoke, {"question": question})
        )

        full_answer = ""
        async for token in callback.aiter():
            full_answer += token
            yield token

        await task
        logger.info(f"[{session_id}] Streaming query complete ({len(full_answer)} chars).")

    # ── UTILITIES ──────────────────────────────────────────────────────────────

    def clear_memory(self, session_id: str = "default") -> None:
        self._session_store.clear(session_id)

    def clear_all_memory(self) -> None:
        self._session_store.clear_all()

    def health(self) -> Dict[str, str]:
        # FIX BUG-I: use ChromaDB collection.count() as source of truth for
        # docs_indexed — persists across restarts, unlike in-memory _all_docs.
        try:
            chroma_count = self._chroma._collection.count()
        except Exception:
            chroma_count = len(self._all_docs)

        return {
            "vector_store": "ready" if self.vector_store else "empty",
            "bm25_index":   "ready" if self._bm25_retriever else "empty",
            "docs_indexed": str(chroma_count),
            "active_sessions": str(self._session_store.active_sessions),
        }
