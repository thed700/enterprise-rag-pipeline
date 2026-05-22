"""
engine.py — AuraRAG Core Engine v3.3
Author: Akmal Raxmatov (github: thed700)

BUG FIXES in v3.3 (on top of v3.2.0):

  BUG-Y: CrossEncoderReranker was initialized and documented but completely
          bypassed in query() and stream_query() — both used
          _build_hybrid_retriever() directly, feeding raw hybrid results
          straight to the LLM.  The reranker only ran in the standalone
          retrieve() method, which no API endpoint calls.
          Fixed: introduced RerankedRetriever, a BaseRetriever subclass that
          wraps hybrid search + cross-encoder reranking in one object.
          _build_reranked_retriever(top_k) constructs it and is now used
          everywhere a retriever is passed to ConversationalRetrievalChain.

  BUG-Z: CrossEncoderReranker.arerank() called the deprecated
          asyncio.get_event_loop() inside an already-running event loop.
          Python 3.10+ emits a DeprecationWarning; a future Python release
          will raise RuntimeError.
          Fixed: replaced with asyncio.get_running_loop().

  BUG-AB: CrossEncoderReranker._executor (a ThreadPoolExecutor) was never
          shut down.  Every hot-reload or graceful server restart leaked OS
          threads until the process exited.
          Fixed: added CrossEncoderReranker.shutdown() and RAGEngine.shutdown();
          the FastAPI lifespan cleanup block calls engine.shutdown().

  BUG-AC: SessionMemoryStore._evict_stale() used the module-level constant
          SESSION_TTL_MINUTES = 60 instead of settings.SESSION_TTL_MINUTES,
          so setting SESSION_TTL_MINUTES in .env had no effect at runtime.
          Fixed: _evict_stale() now calls get_settings().SESSION_TTL_MINUTES.

  BUG-AE: Source snippet truncation in query() was hardcoded to [:300].
          Fixed: reads settings.SOURCE_SNIPPET_LEN so the value is tunable
          via .env without code changes.

Retained from v3.2.0:
  BUG-S: top_k forwarded from API -> engine -> reranker.
  BUG-U: SessionMemoryStore.clear() removes _last_access entry.
  BUG-V: stream_query() propagates chain exceptions to SSE error frames.
  BUG-W: TextLoader uses UTF-8 + autodetect_encoding.
  BUG-X: _seen_hashes persisted in BM25 pickle; atomic pickle writes.
"""

import asyncio
import hashlib
import logging
import os
import pickle
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import chromadb

from langchain_community.chains import ConversationalRetrievalChain # FIX BUG-A
from langchain.memory import ConversationBufferWindowMemory         # FIX BUG-K
from langchain.retrievers.ensemble import EnsembleRetriever         # FIX BUG-B
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

from app.constants import PROVIDER_MODELS, validate_provider_config
from app.utils import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Re-export so any legacy code that imported from engine still works
__all__ = ["RAGEngine", "build_llm", "PROVIDER_MODELS", "validate_provider_config"]

# Kept for import compatibility and tests — NOT used for actual eviction logic
# after the BUG-AC fix.  _evict_stale() reads settings.SESSION_TTL_MINUTES.
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
            f"Re-ranked {len(documents)} docs -> top {top_k}. "
            f"Best score: {ranked[0][0]:.4f}"
        )
        return [doc for _, doc in ranked[:top_k]]

    async def arerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Non-blocking rerank: runs the CPU-bound inference in a thread pool.

        BUG-Z fix: replaced deprecated asyncio.get_event_loop() with
        asyncio.get_running_loop() — the former raises DeprecationWarning in
        Python 3.10+ when called from within a running event loop.
        """
        loop = asyncio.get_running_loop()   # BUG-Z fix
        return await loop.run_in_executor(
            self._executor, self.rerank, query, documents, top_k
        )

    def shutdown(self) -> None:
        """BUG-AB fix: shut down the thread pool to avoid OS thread leaks.

        Call this once during application teardown (e.g. FastAPI lifespan
        cleanup).  wait=False lets the process exit without blocking on any
        in-flight rerank tasks.
        """
        self._executor.shutdown(wait=False)
        logger.debug("CrossEncoderReranker executor shut down.")


# ─────────────────────────────────────────────
# RERANKED RETRIEVER  (BUG-Y fix)
# ─────────────────────────────────────────────

class RerankedRetriever(BaseRetriever):
    """
    BUG-Y fix: LangChain-compatible retriever that applies cross-encoder
    re-ranking after hybrid search in one step.

    Previously the CrossEncoderReranker was only called from retrieve(), a
    standalone helper that no API endpoint actually invokes.  Both query() and
    stream_query() passed _build_hybrid_retriever() directly to
    ConversationalRetrievalChain, completely bypassing the reranker.

    By packaging hybrid search + reranking into a single BaseRetriever
    subclass the reranker is applied automatically whenever the chain fetches
    context documents, with no changes required to the chain setup.
    """

    # Pydantic v2 fields — arbitrary_types_allowed is required because
    # EnsembleRetriever and CrossEncoderReranker are not Pydantic models.
    hybrid_retriever: Any
    cross_encoder: Any
    top_k: int = 5

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> List[Document]:
        candidates = self.hybrid_retriever.invoke(query)
        return self.cross_encoder.rerank(query, candidates, top_k=self.top_k)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> List[Document]:
        candidates = self.hybrid_retriever.invoke(query)
        return await self.cross_encoder.arerank(query, candidates, top_k=self.top_k)


# ─────────────────────────────────────────────
# SESSION MEMORY STORE
# ─────────────────────────────────────────────

class SessionMemoryStore:
    """
    Per-session windowed conversation memory.
    Evicts sessions idle longer than settings.SESSION_TTL_MINUTES.
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
        """
        BUG-U fix: also remove _last_access so the session is no longer
        counted as active and can be re-created fresh immediately.
        """
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            del self._sessions[session_id]
            self._last_access.pop(session_id, None)
            logger.info(f"Memory cleared for session {session_id!r}.")

    def clear_all(self) -> None:
        self._sessions.clear()
        self._last_access.clear()

    def _evict_stale(self) -> None:
        # BUG-AC fix: read the TTL from settings at call time so that
        # SESSION_TTL_MINUTES in .env takes effect.  The old code used the
        # module-level constant SESSION_TTL_MINUTES = 60, which was never
        # overridable via environment variables.
        ttl_minutes = get_settings().SESSION_TTL_MINUTES
        cutoff = time.monotonic() - ttl_minutes * 60
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
    AuraRAG Enterprise Engine v3.3 — LLM-Agnostic.

    Pipeline:
      Ingest  -> Chunk -> Deduplicate -> Embed -> Store (ChromaDB + BM25)
      Query   -> Hybrid Search -> Cross-Encoder Re-rank -> LLM -> Stream

    Concurrency:
      Safe for --workers 1. Per-session memory via SessionMemoryStore.
      BM25 rebuild and ChromaDB writes are synchronous; wrap in asyncio.to_thread()
      at the API layer for non-blocking behaviour.
    """

    def __init__(self) -> None:
        logger.info("Initialising AuraRAG Engine v3.3...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
        )

        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

        self._bm25_pickle_path = pathlib.Path(settings.CHROMA_PERSIST_DIR) / "bm25.pkl"

        _store = Chroma(
            client=self.chroma_client,
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=self.embeddings,
        )
        self.vector_store: Optional[Chroma] = (
            _store if _store._collection.count() > 0 else None
        )
        self._chroma = _store

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        self.reranker = CrossEncoderReranker()
        self._session_store = SessionMemoryStore(window_k=20)

        self._seen_hashes: set[str] = set()
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
                if isinstance(saved, dict):
                    self._bm25_retriever = saved["retriever"]
                    self._all_docs = saved.get("docs", [])
                    # BUG-X fix: restore _seen_hashes from saved set if present;
                    # fall back to rebuilding from docs for legacy pickles.
                    self._seen_hashes = saved.get(
                        "hashes",
                        {_content_hash(d) for d in self._all_docs},
                    )
                else:
                    # Legacy pickle (bare BM25Retriever from v3.0.0)
                    self._bm25_retriever = saved
                logger.info(f"BM25 index restored ({len(self._all_docs)} docs).")
            except Exception as e:
                logger.warning(f"Could not restore BM25 index: {e}. Will rebuild on next ingest.")

    def _save_bm25_to_disk(self) -> None:
        """
        Atomic write: pickle to a .tmp file then rename so a mid-write crash
        never leaves a corrupt pickle on disk.
        """
        try:
            self._bm25_pickle_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._bm25_pickle_path.with_suffix(".pkl.tmp")
            # BUG-X fix: persist _seen_hashes explicitly alongside docs.
            payload = {
                "retriever": self._bm25_retriever,
                "docs":      self._all_docs,
                "hashes":    self._seen_hashes,
            }
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f)
            os.replace(tmp_path, self._bm25_pickle_path)  # atomic on POSIX
            logger.info(f"BM25 index saved ({len(self._all_docs)} docs).")
        except Exception as e:
            logger.warning(f"Could not persist BM25 index: {e}.")

    # ── INGESTION ──────────────────────────────────────────────────────────────

    def ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
        logger.info(f"Ingesting {len(documents)} raw document(s)...")
        chunks = self.text_splitter.split_documents(documents)

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
            self.vector_store = self._chroma

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

    def _build_reranked_retriever(self, top_k: int = 5) -> RerankedRetriever:
        """
        BUG-Y fix: return a RerankedRetriever that combines hybrid search and
        cross-encoder reranking in one BaseRetriever-compatible object.

        Use this everywhere a retriever is passed to ConversationalRetrievalChain
        so the reranker is active on every query, not only in retrieve().
        """
        return RerankedRetriever(
            hybrid_retriever=self._build_hybrid_retriever(),
            cross_encoder=self.reranker,
            top_k=top_k,
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Standalone hybrid search + cross-encoder re-rank (no LLM)."""
        hybrid = self._build_hybrid_retriever()
        candidates = hybrid.invoke(query)
        return self.reranker.rerank(query, candidates, top_k=top_k)

    # ── SYNCHRONOUS QUERY ─────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        provider: str,
        model: str,
        api_key: str,
        session_id: str = "default",
        top_k: int = 5,           # BUG-S fix: top_k forwarded from API layer
    ) -> Dict[str, Any]:
        if self.vector_store is None:
            return {
                "answer": "No documents ingested yet. Upload files first.",
                "sources": [],
                "chat_history": [],
                "session_id": session_id,
            }

        memory = self._session_store.get(session_id)
        llm = build_llm(provider=provider, model=model, api_key=api_key, temperature=0.0)

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            # BUG-Y fix: RerankedRetriever applies cross-encoder reranking
            # inside the chain; previously used _build_hybrid_retriever()
            # which completely bypassed the reranker.
            retriever=self._build_reranked_retriever(top_k=top_k),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": STRICT_PROMPT},
            return_source_documents=True,
            verbose=False,
        )

        logger.info(f"[{session_id}] Query via {provider}/{model} (top_k={top_k}): '{question[:80]}'")
        result = chain.invoke({"question": question})

        # BUG-AE fix: snippet length now comes from settings.SOURCE_SNIPPET_LEN
        # instead of being hardcoded to 300.
        snippet_len = get_settings().SOURCE_SNIPPET_LEN
        sources = [
            {"content": doc.page_content[:snippet_len], "metadata": doc.metadata}
            for doc in result.get("source_documents", [])[:top_k]
        ]

        return {
            "answer": result["answer"],
            "sources": sources,
            "chat_history": [m.content for m in memory.chat_memory.messages],
            "session_id": session_id,
        }

    # ── TRUE SSE STREAMING ────────────────────────────────────────────────────

    async def stream_query(
        self,
        question: str,
        provider: str,
        model: str,
        api_key: str,
        session_id: str = "default",
        top_k: int = 5,           # BUG-S fix
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that yields LLM tokens as they arrive.
        Consumed by the FastAPI /query/stream SSE endpoint.

        BUG-Y fix: now uses _build_reranked_retriever() so cross-encoder
        reranking is active during streaming retrieval.

        BUG-V fix: exceptions from the background chain task are properly
        propagated into the generator so the SSE layer emits an error frame
        instead of hanging indefinitely.
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
            # BUG-Y fix: use RerankedRetriever so reranking is active
            retriever=self._build_reranked_retriever(top_k=top_k),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": STRICT_PROMPT},
            return_source_documents=False,
            verbose=False,
        )

        # BUG-V fix: wrap chain.invoke in a task; propagate exceptions into
        # the generator rather than silently swallowing them.
        task: asyncio.Task = asyncio.create_task(
            asyncio.to_thread(chain.invoke, {"question": question})
        )

        full_answer = ""
        try:
            async for token in callback.aiter():
                full_answer += token
                yield token
        except Exception as gen_err:
            # Cancel the background task if the consumer bails early
            task.cancel()
            raise gen_err

        # Re-raise any exception the chain raised (BUG-V)
        try:
            await task
        except Exception as chain_err:
            logger.exception(f"[{session_id}] Chain error in stream_query")
            raise chain_err

        logger.info(f"[{session_id}] Streaming query complete ({len(full_answer)} chars).")

    # ── UTILITIES ──────────────────────────────────────────────────────────────

    def clear_memory(self, session_id: str = "default") -> None:
        self._session_store.clear(session_id)

    def clear_all_memory(self) -> None:
        self._session_store.clear_all()

    def shutdown(self) -> None:
        """
        BUG-AB fix: release all engine resources during application teardown.
        Call from the FastAPI lifespan cleanup block (after `yield`) to avoid
        OS thread leaks from the CrossEncoderReranker's ThreadPoolExecutor.
        """
        self.reranker.shutdown()
        logger.info("AuraRAG Engine resources released.")

    def health(self) -> Dict[str, str]:
        try:
            chroma_count = self._chroma._collection.count()
        except Exception:
            chroma_count = len(self._all_docs)

        bm25_count = len(self._all_docs) if self._bm25_retriever else 0

        return {
            "vector_store":    "ready" if self.vector_store else "empty",
            "bm25_index":      "ready" if self._bm25_retriever else "empty",
            "docs_indexed":    str(chroma_count),
            "bm25_docs":       str(bm25_count),
            "active_sessions": str(self._session_store.active_sessions),
        }
