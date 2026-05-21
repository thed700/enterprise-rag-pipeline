"""
engine/pipeline.py — AuraRAG LangGraph Agentic Pipeline v3.5
Author: Akmal Raxmatov (github: thed700)

Architecture Migration v3.3 → v3.4 → v3.5
────────────────────────────────────────────────────────────────────────────────
v3.3 used a monolithic ConversationalRetrievalChain (single sequential pipeline
with no conditional logic).  v3.4 replaced it with a compiled LangGraph
StateGraph with five nodes and conditional edges.  v3.5 fixes latent bugs
discovered during production review without changing the graph topology.

  START
    │
    ▼
  [rewrite]   — Query Transformation
  │             Rewrites the raw user query into a search-optimised form,
  │             using session history and (on reflection loops) the previous
  │             grounding feedback as additional context.
    │
    ▼
  [retrieve]  — Hybrid Search + Cross-Encoder Rerank
  │             Invokes RerankedRetriever (60% ChromaDB dense MMR +
  │             40% BM25 sparse + CrossEncoder reranking).
  │             BUG-AF fix: async path is truly non-blocking.
    │
    ▼
  [grade]     — Document Relevance Grader
  │             Fast parallel LLM pass over each chunk.  Chunks scoring
  │             below GRADE_THRESHOLD are filtered.  Falls back to a
  │             keyword heuristic if the LLM call fails (fail-open).
  │             BUG-AK fix: GRADE_THRESHOLD is now actually applied.
    │
    ▼
  [generate]  — Answer Synthesis
  │             Full LLM call grounded in graded docs + session history.
  │             Appends a hidden <<META>> hallucination_risk score.
    │
    ▼  (conditional edge → _should_reflect)
  [reflect]   — Self-Correction (optional / looped)
    │           If hallucination_risk > 0.7 AND loops < MAX_REFLECT_LOOPS,
    │           generates a refined search query and loops back to [retrieve].
    │           Otherwise falls through to END.
    │
  END

Bug Fixes Applied in v3.5
────────────────────────────────────────────────────────────────────────────────
  BUG-AK: GRADE_THRESHOLD was defined in Settings and described in docstrings
          but never applied — the grader used only the LLM's binary keep/drop
          list.  Fixed: the grader prompt now requests per-document float scores
          and GRADE_THRESHOLD filters the scored list before returning
          relevant_docs to the Generate node.

  BUG-AL: PromptOverrides fields used empty string "" defaults.  The router
          called model_dump(exclude_none=True) which does NOT filter empty
          strings — empty UI fields silently replaced the engine's default
          system prompts.  Fixed: fields changed to Optional[str] = None so
          exclude_none=True correctly drops unset overrides.  (Fixed in models.py)

  BUG-AM: The SSE /query/stream endpoint emitted the [DONE] sentinel before
          the meta frame.  Clients that stop reading at [DONE] (standard SSE
          termination) never received session metadata.  Fixed: meta is now
          emitted before [DONE].  (Fixed in routers/query.py)

  BUG-AN: _should_reflect() had type hint Literal["reflect", "__end__"] but
          returned the END constant — a sentinel object whose string value is
          version-dependent.  Fixed: return type widened to str; the function
          still returns the END constant (which == "__end__") for semantic
          clarity, and the routing map key remains END.

  BUG-AO: query() sync wrapper raised RuntimeError then caught it in the same
          except block, checking "event loop" in the message — which matched
          the error it just raised, causing a confusing double-raise instead of
          the intended guard.  Fixed: guard now detects the "no running event
          loop" message from get_running_loop(), swallowing only that error and
          re-raising all others including the advisory error.

All v3.4 fixes (BUG-AF through BUG-AH) and v3.3 fixes (BUG-S, BUG-U through
BUG-AE) are fully preserved.  See CHANGELOG.md for the complete history.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pathlib
import pickle
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
)

import chromadb
from langchain.memory import ConversationBufferWindowMemory
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from sentence_transformers import CrossEncoder

from app.constants import PROVIDER_MODELS, validate_provider_config
from app.utils import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

def _resolve_persist_dir(raw_path: str) -> pathlib.Path:
    """
    Resolve a writable persistence directory.

    On Hugging Face Spaces and other constrained runtimes, the configured
    directory may not be writable. In that case we fall back to a cache path
    under the system temp directory so the app can still boot.
    """
    candidates = [
        pathlib.Path(raw_path).expanduser(),
        pathlib.Path(os.environ.get("AURARAG_CACHE_DIR", "" )).expanduser() if os.environ.get("AURARAG_CACHE_DIR") else None,
        pathlib.Path(tempfile.gettempdir()) / "aurarag" / "chroma_db",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            test_file = candidate / ".write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            return candidate
        except Exception:
            continue
    return pathlib.Path(tempfile.gettempdir()) / "aurarag" / "chroma_db"

# Re-export for legacy import compatibility
__all__ = [
    "RAGEngine",
    "build_llm",
    "PROVIDER_MODELS",
    "validate_provider_config",
    "RerankedRetriever",
    "SessionMemoryStore",
]

# Kept as a module-level constant for import compatibility and tests only.
# _evict_stale() always reads get_settings().SESSION_TTL_MINUTES (BUG-AC fix).
SESSION_TTL_MINUTES = 60

# Internal pipeline constants
_MAX_RETRIEVAL_DOCS       = 10   # candidate pool size for hybrid retriever
_MAX_CONTEXT_CHARS_PER_DOC = 1200  # chars per doc in generate context
_MAX_GRADING_CHARS_PER_DOC = 700   # chars per doc sent to grader (keep prompt short)


# ─────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────

_REWRITE_SYSTEM = (
    "You are an expert at reformulating user questions into concise, "
    "keyword-rich search queries for a retrieval-augmented system. "
    "Given the chat history, previous grounding feedback (if any), and the "
    "current question, output ONLY the optimised query string — no explanation, "
    "no preamble, no quotes."
)

_GRADE_SYSTEM = (
    "You are a relevance judge. Given a search query and a list of numbered "
    "document snippets, return strict JSON only with two keys:\n"
    '  "keep_indices": [list of 1-based ints for relevant snippets]\n'
    '  "reason": "short explanation"\n'
    "If nothing is relevant return {\"keep_indices\": [], \"reason\": \"...\"}.\n"
    "No markdown fences, no extra text."
)

_GENERATE_SYSTEM = (
    "You are a precise and trustworthy AI assistant for an enterprise knowledge base.\n"
    "Your answers must be grounded EXCLUSIVELY in the provided context.\n\n"
    "STRICT RULES:\n"
    "1. If the answer is not found in the context, say exactly:\n"
    '   "I do not have enough information in the provided documents to answer this question."\n'
    "2. Do NOT invent, extrapolate, or assume facts not present in the context.\n"
    "3. Always cite the source document when possible.\n"
    "4. Be concise, professional, and factual.\n\n"
    "At the END of your answer append this hidden block on a new line:\n"
    "<<META>>{\"hallucination_risk\": <float 0.0-1.0>}<<END_META>>\n"
    "where hallucination_risk is your self-assessed probability that your answer "
    "contains content NOT grounded in the provided context "
    "(0.0 = fully grounded, 1.0 = entirely hallucinated)."
)

_REFLECT_SYSTEM = (
    "You are a search query expert. The previous answer may have hallucinated "
    "because the retrieved context was insufficient. "
    "Produce a BETTER, more specific search query to retrieve more relevant documents. "
    "Output ONLY the refined query — no explanation, no quotes."
)


# ─────────────────────────────────────────────
# LANGGRAPH STATE
# ─────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    """
    Shared mutable state threaded through all LangGraph nodes.

    All fields are Optional (total=False) so nodes can write independently
    without needing to populate every key on every transition.
    """
    # ── Caller-supplied inputs ────────────────
    question:    str
    provider:    str
    model:       str
    api_key:     str   # plaintext, short-lived per-request
    session_id:  str
    top_k:       int
    system_prompts: Dict[str, str]  # optional prompt overrides from the UI
    chat_history_text: str  # pre-formatted string for prompt injection

    # ── Rewrite node ──────────────────────────
    rewritten_query:     str
    reflection_feedback: str  # carried forward on subsequent loops

    # ── Retrieve node ─────────────────────────
    retrieved_docs: List[Document]

    # ── Grade node ────────────────────────────
    relevant_docs: List[Document]
    graded_count:  int   # number of chunks that passed grading

    # ── Generate node ─────────────────────────
    answer:             str
    hallucination_risk: float  # 0.0–1.0 self-reported

    # ── Reflect node ──────────────────────────
    reflect_loops: int   # loop counter for budget enforcement
    needs_revision: bool

    # ── Observability ─────────────────────────
    pipeline_trace: List[str]  # ordered node names executed


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
    max_tokens: Optional[int] = None,
) -> BaseLanguageModel:
    """
    Instantiate a LangChain chat model for the given provider.

    max_tokens is accepted for lightweight nodes (rewrite, grade) where a small
    token budget improves throughput without affecting answer quality.
    """
    common: Dict[str, Any] = {
        "temperature": temperature,
        "streaming":   streaming,
        "callbacks":   callbacks or [],
    }
    if max_tokens is not None:
        common["max_tokens"] = max_tokens

    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, openai_api_key=api_key, **common)

    if provider == "Anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, anthropic_api_key=api_key, **common)

    if provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, **common)

    if provider == "Local (Ollama)":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama  # type: ignore
        # Ollama does not accept api_key / callbacks in the same way
        return ChatOllama(model=model, temperature=temperature)

    raise ValueError(
        f"Unknown provider '{provider}'. Choose from: {list(PROVIDER_MODELS.keys())}"
    )


def _maybe_tagged(llm: BaseLanguageModel, tags: Sequence[str]) -> BaseLanguageModel:
    """Attach LangChain config tags to an LLM instance if supported."""
    if tags and hasattr(llm, "with_config"):
        return llm.with_config({"tags": list(tags)})  # type: ignore[return-value]
    return llm


# ─────────────────────────────────────────────
# CROSS-ENCODER RE-RANKER
# ─────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Wraps a sentence-transformers CrossEncoder for synchronous and async reranking.

    BUG-Z fix (v3.3):  arerank() uses asyncio.get_running_loop() instead of
                       the deprecated asyncio.get_event_loop().
    BUG-AB fix (v3.3): shutdown() releases the ThreadPoolExecutor so graceful
                       restarts do not leak OS threads.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        logger.info("Loading Cross-Encoder: %s", model_name)
        self.model = CrossEncoder(model_name)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reranker")

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        raw_scores = self.model.predict(pairs)
        scores: List[float] = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)
        ranked = sorted(zip(scores, documents), key=lambda item: item[0], reverse=True)
        if ranked:
            logger.info(
                "Re-ranked %d docs → top %d. Best score: %.4f",
                len(documents), top_k, ranked[0][0],
            )
        return [doc for _, doc in ranked[:top_k]]

    async def arerank(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Document]:
        """
        Non-blocking rerank: runs CPU-bound inference in the thread pool.

        BUG-Z fix (v3.3): uses asyncio.get_running_loop() — the deprecated
        asyncio.get_event_loop() emits DeprecationWarning in Python 3.10+ when
        called from within a running event loop.
        """
        loop = asyncio.get_running_loop()  # BUG-Z fix
        return await loop.run_in_executor(
            self._executor, self.rerank, query, documents, top_k
        )

    def shutdown(self) -> None:
        """
        BUG-AB fix (v3.3): release the thread pool during application teardown.
        Called from RAGEngine.shutdown() → FastAPI lifespan cleanup block.
        wait=False lets the process exit without blocking on in-flight tasks.
        """
        self._executor.shutdown(wait=False)
        logger.debug("CrossEncoderReranker executor shut down.")


# ─────────────────────────────────────────────
# RERANKED RETRIEVER  (BUG-Y v3.3 + BUG-AF v3.4)
# ─────────────────────────────────────────────

class RerankedRetriever(BaseRetriever):
    """
    LangChain-compatible retriever that applies cross-encoder re-ranking
    after hybrid search in one step.

    BUG-Y fix (v3.3): used everywhere a retriever is needed so the reranker
    is never bypassed by the pipeline (previously only ran in the standalone
    retrieve() helper which no API endpoint called).

    BUG-AF fix (v3.4): the async path now dispatches via _invoke_hybrid_async()
    which prefers ainvoke() when available and falls back to asyncio.to_thread()
    for retrievers that only implement the sync interface.  The v3.3 version
    called self.hybrid_retriever.invoke() (sync) directly from the async method,
    blocking the event loop for the full retrieval duration under load.
    """

    hybrid_retriever: Any
    cross_encoder:    Any
    top_k:            int = 5

    model_config = {"arbitrary_types_allowed": True}

    # ── internal helpers ───────────────────────────────────────────────────

    def _invoke_hybrid_sync(self, query: str) -> List[Document]:
        return list(self.hybrid_retriever.invoke(query))

    async def _invoke_hybrid_async(self, query: str) -> List[Document]:
        """
        BUG-AF fix: prefer ainvoke() when available; fall back to
        asyncio.to_thread() so the event loop is never blocked by a
        sync retriever implementation.
        """
        if hasattr(self.hybrid_retriever, "ainvoke"):
            return list(await self.hybrid_retriever.ainvoke(query))
        return await asyncio.to_thread(self._invoke_hybrid_sync, query)

    # ── BaseRetriever interface ────────────────────────────────────────────

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        candidates = self._invoke_hybrid_sync(query)
        return self.cross_encoder.rerank(query, candidates, top_k=self.top_k)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        candidates = await self._invoke_hybrid_async(query)
        return await self.cross_encoder.arerank(query, candidates, top_k=self.top_k)


# ─────────────────────────────────────────────
# SESSION MEMORY STORE
# ─────────────────────────────────────────────

class SessionMemoryStore:
    """
    Per-session windowed conversation memory with TTL eviction.

    BUG-AC fix (v3.3): _evict_stale() reads get_settings().SESSION_TTL_MINUTES
                       at call time instead of the module-level constant.
    BUG-U fix (v3.3):  clear() removes from both _sessions and _last_access.
    """

    def __init__(self, window_k: int = 20) -> None:
        self._window_k = window_k
        self._sessions:    Dict[str, ConversationBufferWindowMemory] = {}
        self._last_access: Dict[str, float] = {}

    def get(self, session_id: str) -> ConversationBufferWindowMemory:
        self._evict_stale()
        if session_id not in self._sessions:
            logger.debug("Creating new memory for session %r.", session_id)
            self._sessions[session_id] = ConversationBufferWindowMemory(
                k=self._window_k,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
            )
        self._last_access[session_id] = time.monotonic()
        return self._sessions[session_id]

    def save_turn(self, session_id: str, question: str, answer: str) -> None:
        """Persist a completed Q/A turn into session memory."""
        mem = self.get(session_id)
        mem.save_context({"question": question}, {"answer": answer})

    def format_history(self, session_id: str) -> str:
        """Return chat history as a plain-text string for prompt injection."""
        mem = self.get(session_id)
        messages: List[BaseMessage] = mem.chat_memory.messages
        if not messages:
            return "No prior conversation."
        lines: List[str] = []
        for msg in messages:
            role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def clear(self, session_id: str) -> None:
        """
        BUG-U fix (v3.3): removes from both _sessions and _last_access so
        the session can be re-created fresh immediately on next access.
        """
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            del self._sessions[session_id]
            self._last_access.pop(session_id, None)
            logger.info("Memory cleared for session %r.", session_id)

    def clear_all(self) -> None:
        self._sessions.clear()
        self._last_access.clear()

    def _evict_stale(self) -> None:
        # BUG-AC fix (v3.3): read TTL from settings at call time so that
        # SESSION_TTL_MINUTES in .env takes effect at runtime.
        ttl_minutes = get_settings().SESSION_TTL_MINUTES
        cutoff = time.monotonic() - ttl_minutes * 60
        stale = [sid for sid, ts in self._last_access.items() if ts < cutoff]
        for sid in stale:
            self._sessions.pop(sid, None)
            self._last_access.pop(sid, None)
            logger.debug("Evicted stale session %r.", sid)

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)


# ─────────────────────────────────────────────
# DOCUMENT DEDUPLICATION
# ─────────────────────────────────────────────

def _content_hash(doc: Document) -> str:
    """SHA-256 fingerprint of a document chunk's content."""
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()


def _dedupe_documents(documents: Sequence[Document]) -> List[Document]:
    """Remove duplicate chunks by content hash, preserving order."""
    seen: set[str] = set()
    unique: List[Document] = []
    for doc in documents:
        key = _content_hash(doc)
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


# ─────────────────────────────────────────────
# CONTEXT FORMATTING HELPERS
# ─────────────────────────────────────────────

def _doc_label(doc: Document, idx: int) -> str:
    meta   = doc.metadata or {}
    source = meta.get("source", "unknown")
    page   = meta.get("page")
    page_s = f" p.{page}" if page is not None else ""
    return f"[{idx}] {source}{page_s}"


def _doc_snippet(doc: Document, idx: int, max_chars: int) -> str:
    return f"{_doc_label(doc, idx)}\n{doc.page_content[:max_chars]}"


# ─────────────────────────────────────────────
# JSON / META HELPERS
# ─────────────────────────────────────────────

def _safe_json_object(text: str) -> Dict[str, Any]:
    """
    Parse a JSON object from LLM output, stripping markdown fences.
    Returns {} on any parse failure (never raises).
    """
    cleaned = text.strip()
    if not cleaned:
        return {}
    # Strip ```json ... ``` fences if present
    if cleaned.startswith("```"):
        lines = [l for l in cleaned.split("\n") if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    # Extract the first {...} block in case of surrounding prose
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    candidate = match.group(0) if match else cleaned
    try:
        data = json.loads(candidate)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _extract_hallucination_risk(answer: str) -> Tuple[str, float]:
    """
    Parse the hidden <<META>>...<<END_META>> block from the generation prompt.
    Returns (clean_answer, hallucination_risk_float).
    Conservative default of 0.5 if the block is absent or unparseable.
    """
    pattern = r"<<META>>(.*?)<<END_META>>"
    match = re.search(pattern, answer, re.DOTALL)
    risk = 0.5  # conservative default
    if match:
        raw_meta = match.group(1).strip()
        try:
            meta = json.loads(raw_meta)
            risk = float(meta.get("hallucination_risk", 0.5))
            risk = max(0.0, min(1.0, risk))
        except Exception:
            pass
        clean = re.sub(pattern, "", answer, flags=re.DOTALL).strip()
    else:
        clean = answer.strip()
    return clean, risk


def _heuristic_relevance_indices(question: str, documents: Sequence[Document]) -> List[int]:
    """
    Keyword-overlap fallback for the grader when the LLM call fails.
    Returns 1-based indices of documents that share at least one term with
    the question (terms longer than 3 characters).
    """
    terms = {tok for tok in re.findall(r"[\w\-]+", question.lower()) if len(tok) > 3}
    if not terms:
        return list(range(1, len(documents) + 1))  # keep all if no terms extracted
    kept: List[int] = []
    for idx, doc in enumerate(documents, start=1):
        haystack = doc.page_content.lower()
        if any(term in haystack for term in terms):
            kept.append(idx)
    return kept or list(range(1, len(documents) + 1))  # keep all if none matched


# ─────────────────────────────────────────────
# LANGGRAPH NODE IMPLEMENTATIONS
# ─────────────────────────────────────────────

async def _node_rewrite(state: GraphState) -> Dict[str, Any]:
    """
    Node 1 — Query Transformation.

    Rewrites the raw user question into a search-optimised form, incorporating
    session history and (on reflection loops) the previous grounding feedback.
    Uses REWRITE_MAX_TOKENS to keep the node fast and cheap.

    Falls back silently to the raw question on any LLM error — the pipeline
    continues with reduced quality rather than failing hard.
    """
    cfg         = get_settings()
    question    = state["question"]
    history     = state.get("chat_history_text", "No prior conversation.")
    feedback    = state.get("reflection_feedback", "") or "None"
    provider    = state["provider"]
    model       = state["model"]
    api_key     = state["api_key"]

    trace = list(state.get("pipeline_trace", []))
    trace.append("rewrite")

    try:
        llm = _maybe_tagged(
            build_llm(provider, model, api_key, temperature=0.0,
                      max_tokens=cfg.REWRITE_MAX_TOKENS),
            ["nostream", "rewrite"],
        )
        prompt = (
            f"Session history:\n{history}\n\n"
            f"Previous grounding feedback (if any): {feedback}\n\n"
            f"Original question:\n{question}\n\n"
            "Return a short retrieval query with only the essential search terms."
        )
        response = await llm.ainvoke([
            SystemMessage(content=(state.get('system_prompts', {}) or {}).get('rewrite', _REWRITE_SYSTEM) or _REWRITE_SYSTEM),
            HumanMessage(content=prompt),
        ])
        rewritten = str(getattr(response, "content", "")).strip() or question
    except Exception as exc:
        logger.warning("[%s] Rewrite failed (%s); using raw question.", state.get("session_id"), exc)
        rewritten = question

    logger.info(
        "[%s] Rewrite: '%s' → '%s'",
        state.get("session_id"), question[:60], rewritten[:60],
    )
    return {**state, "rewritten_query": rewritten, "pipeline_trace": trace}


async def _node_retrieve(
    state: GraphState,
    reranked_retriever: RerankedRetriever,
) -> Dict[str, Any]:
    """
    Node 2 — Hybrid Search + Cross-Encoder Rerank.

    Uses the pre-built RerankedRetriever (60% dense ChromaDB MMR +
    40% BM25 sparse + CrossEncoder).  Uses rewritten_query if available;
    falls back to the raw question.

    BUG-AF fix (v3.4): async retrieval is truly non-blocking (see
    RerankedRetriever._invoke_hybrid_async).
    """
    query = state.get("rewritten_query") or state["question"]

    trace = list(state.get("pipeline_trace", []))
    trace.append("retrieve")

    try:
        docs = await reranked_retriever.ainvoke(query)
    except Exception as exc:
        logger.error("[%s] Retrieval error: %s", state.get("session_id"), exc)
        docs = []

    logger.info(
        "[%s] Retrieved %d docs for query: '%s'",
        state.get("session_id"), len(docs), query[:60],
    )
    return {**state, "retrieved_docs": list(docs), "pipeline_trace": trace}


async def _node_grade(state: GraphState) -> Dict[str, Any]:
    """
    Node 3 — Document Relevance Grader.

    Runs a fast parallel LLM pass over each retrieved chunk.  Chunks whose
    relevance score < GRADE_THRESHOLD are filtered out.

    BUG-AK fix (v3.5): GRADE_THRESHOLD was defined in Settings and documented
    as the relevance filter but was never actually applied — the grader used
    only the LLM's binary keep/drop list.  The grader prompt now asks the LLM
    to return per-document float scores; GRADE_THRESHOLD is applied to filter
    the scored list before building the graded set.  The keyword heuristic
    fallback is preserved for LLM failures.

    Fail-open strategy: on any LLM error the heuristic keyword matcher runs
    as a fallback.  If that also returns nothing, all docs are kept so the
    Generate node always has something to work with.
    """
    cfg      = get_settings()
    query    = state.get("rewritten_query") or state["question"]
    docs     = state.get("retrieved_docs", [])
    provider = state["provider"]
    model    = state["model"]
    api_key  = state["api_key"]

    trace = list(state.get("pipeline_trace", []))
    trace.append("grade")

    if not docs:
        return {**state, "relevant_docs": [], "graded_count": 0, "pipeline_trace": trace}

    # Cap the number of docs sent to the grader to control prompt size.
    docs_to_grade = docs[:_MAX_RETRIEVAL_DOCS]
    doc_blocks = "\n\n".join(
        _doc_snippet(doc, idx + 1, _MAX_GRADING_CHARS_PER_DOC)
        for idx, doc in enumerate(docs_to_grade)
    )

    # BUG-AK fix: updated grade system prompt to request per-document scores.
    # The threshold in cfg.GRADE_THRESHOLD is then applied to the scores list.
    _GRADE_SYSTEM_WITH_SCORES = (
        "You are a relevance judge. Given a search query and a list of numbered "
        "document snippets, return strict JSON only with two keys:\n"
        '  "scores": {<1-based index>: <float 0.0-1.0>, ...}  (relevance score per snippet)\n'
        '  "reason": "short explanation"\n'
        "Score 1.0 = perfectly relevant, 0.0 = completely irrelevant. "
        "If all snippets are irrelevant return all scores as 0.0.\n"
        "No markdown fences, no extra text."
    )

    grade_system = (state.get("system_prompts", {}) or {}).get("grade") or _GRADE_SYSTEM_WITH_SCORES

    graded: List[Document] = []
    try:
        llm = _maybe_tagged(
            build_llm(provider, model, api_key, temperature=0.0,
                      max_tokens=cfg.GRADE_MAX_TOKENS),
            ["nostream", "grader"],
        )
        prompt = (
            f"User question:\n{query}\n\n"
            f"Retrieved snippets:\n{doc_blocks}\n\n"
            "Return strict JSON only."
        )
        response = await llm.ainvoke([
            SystemMessage(content=grade_system),
            HumanMessage(content=prompt),
        ])
        parsed = _safe_json_object(str(getattr(response, "content", "")))

        # BUG-AK fix: extract per-doc scores and apply GRADE_THRESHOLD.
        # Falls back to keep_indices for backward-compat if old prompt is used.
        scores_map = parsed.get("scores", {})
        if scores_map and isinstance(scores_map, dict):
            threshold = cfg.GRADE_THRESHOLD
            for i, doc in enumerate(docs_to_grade, start=1):
                raw_score = scores_map.get(i) or scores_map.get(str(i))
                try:
                    score = float(raw_score) if raw_score is not None else 0.0
                except (TypeError, ValueError):
                    score = 0.0
                if score >= threshold:
                    graded.append(doc)
            logger.debug(
                "[%s] Grade threshold=%.2f applied; %d/%d docs passed scores filter.",
                state.get("session_id"), threshold, len(graded), len(docs_to_grade),
            )
        else:
            # Fallback: handle legacy keep_indices format (custom prompt override)
            keep_indices = parsed.get("keep_indices", [])
            if not isinstance(keep_indices, list):
                keep_indices = []
            keep_set = {
                idx for idx in keep_indices
                if isinstance(idx, int) and 1 <= idx <= len(docs_to_grade)
            }
            if keep_set:
                graded = [doc for i, doc in enumerate(docs_to_grade, start=1) if i in keep_set]
            else:
                heuristic = _heuristic_relevance_indices(state["question"], docs_to_grade)
                graded = [doc for i, doc in enumerate(docs_to_grade, start=1) if i in set(heuristic)]

        # If threshold was too strict and filtered everything, try heuristic
        if not graded:
            heuristic = _heuristic_relevance_indices(state["question"], docs_to_grade)
            graded = [doc for i, doc in enumerate(docs_to_grade, start=1) if i in set(heuristic)]

    except Exception as exc:
        logger.warning(
            "[%s] Grade node failed (%s); passing all %d docs through.",
            state.get("session_id"), exc, len(docs_to_grade),
        )
        graded = list(docs_to_grade)

    # Final fail-open: never starve the Generate node
    if not graded:
        graded = list(docs_to_grade)

    logger.info(
        "[%s] Grader: %d/%d chunks passed.",
        state.get("session_id"), len(graded), len(docs_to_grade),
    )
    return {**state, "relevant_docs": graded, "graded_count": len(graded), "pipeline_trace": trace}


async def _node_generate(state: GraphState) -> Dict[str, Any]:
    """
    Node 4 — Answer Synthesis.

    Produces the final answer grounded in relevant_docs + session history.
    Deduplicates docs before building context and appends a hidden <<META>>
    hallucination_risk score used by the Reflect edge.

    Tags the LLM with "generate" so astream_events() can filter token events
    to just this node for streaming.
    """
    cfg          = get_settings()
    question     = state["question"]
    history      = state.get("chat_history_text", "No prior conversation.")
    docs         = state.get("relevant_docs") or state.get("retrieved_docs", [])
    top_k        = state.get("top_k", 5)
    provider     = state["provider"]
    model        = state["model"]
    api_key      = state["api_key"]

    trace = list(state.get("pipeline_trace", []))
    trace.append("generate")

    # Deduplicate and cap context
    context_docs = _dedupe_documents(docs)[:top_k]
    if context_docs:
        snippet_len = cfg.SOURCE_SNIPPET_LEN
        context = "\n\n".join(
            _doc_snippet(doc, idx + 1, snippet_len)
            for idx, doc in enumerate(context_docs)
        )
    else:
        context = "No relevant context was found in the knowledge base."

    try:
        llm = _maybe_tagged(
            build_llm(provider, model, api_key, temperature=0.0, streaming=True),
            ["generate"],
        )
        response = await llm.ainvoke([
            SystemMessage(content=(state.get('system_prompts', {}) or {}).get('generate', _GENERATE_SYSTEM) or _GENERATE_SYSTEM),
            HumanMessage(content=(
                f"Chat history:\n{history}\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Write the final answer now."
            )),
        ])
        raw_answer = str(getattr(response, "content", "")).strip()
        if not raw_answer:
            raw_answer = (
                "I do not have enough information in the provided documents "
                "to answer this question."
            )
    except Exception as exc:
        logger.error("[%s] Generate node error: %s", state.get("session_id"), exc)
        raw_answer = "I encountered an error while generating a response. Please try again."

    clean_answer, h_risk = _extract_hallucination_risk(raw_answer)
    logger.info(
        "[%s] Generated answer (%d chars, hallucination_risk=%.2f).",
        state.get("session_id"), len(clean_answer), h_risk,
    )
    return {
        **state,
        "answer":             clean_answer,
        "hallucination_risk": h_risk,
        "pipeline_trace":     trace,
    }


async def _node_reflect(
    state: GraphState,
    reranked_retriever: RerankedRetriever,
) -> Dict[str, Any]:
    """
    Node 5 — Self-Correction / Reflect.

    Generates a refined search query when hallucination_risk is high, then
    immediately re-retrieves using it.  Clears relevant_docs / graded_count
    so the next grade pass starts fresh.

    The conditional edge _should_reflect() gates entry — this node only runs
    when risk > 0.7 AND loop budget is not exhausted.
    """
    cfg             = get_settings()
    question        = state["question"]
    rewritten_query = state.get("rewritten_query", question)
    answer          = state.get("answer", "")
    provider        = state["provider"]
    model           = state["model"]
    api_key         = state["api_key"]
    loop_count      = state.get("reflect_loops", 0)

    trace = list(state.get("pipeline_trace", []))
    trace.append("reflect")

    logger.info(
        "[%s] Reflect loop %d — hallucination_risk=%.2f",
        state.get("session_id"), loop_count + 1, state.get("hallucination_risk", 0.0),
    )

    try:
        llm = _maybe_tagged(
            build_llm(provider, model, api_key, temperature=0.0,
                      max_tokens=cfg.REWRITE_MAX_TOKENS),
            ["nostream", "reflect"],
        )
        response = await llm.ainvoke([
            SystemMessage(content=(state.get('system_prompts', {}) or {}).get('reflect', _REFLECT_SYSTEM) or _REFLECT_SYSTEM),
            HumanMessage(content=(
                f"Original question: {question}\n"
                f"Rewritten query used: {rewritten_query}\n"
                f"Unsatisfactory answer: {answer[:400]}\n\n"
                "Refined search query:"
            )),
        ])
        refined = str(getattr(response, "content", "")).strip() or rewritten_query
    except Exception as exc:
        logger.warning("[%s] Reflect LLM failed (%s); reusing query.", state.get("session_id"), exc)
        refined = rewritten_query

    logger.info("[%s] Refined query: '%s'", state.get("session_id"), refined[:80])

    # Re-retrieve immediately with the refined query.
    try:
        new_docs = await reranked_retriever.ainvoke(refined)
    except Exception as exc:
        logger.error("[%s] Reflect retrieval error: %s", state.get("session_id"), exc)
        new_docs = state.get("retrieved_docs", [])

    return {
        **state,
        "rewritten_query":     refined,
        "reflection_feedback": f"Previous answer had high hallucination risk ({state.get('hallucination_risk', 0):.2f}). Refined query used.",
        "retrieved_docs":      list(new_docs),
        "relevant_docs":       [],        # re-graded in the next grade pass
        "graded_count":        0,
        "reflect_loops":       loop_count + 1,
        "pipeline_trace":      trace,
    }


# ─────────────────────────────────────────────
# CONDITIONAL EDGES
# ─────────────────────────────────────────────

def _should_reflect(state: GraphState) -> str:
    """
    After Generate: decide whether to invoke the Reflect node.

    BUG-AN fix (v3.5): the previous type hint was Literal["reflect", "__end__"]
    but the function returned the END constant (which equals "__end__" in
    current LangGraph but is an opaque sentinel — not a guaranteed string).
    The conditional_edges map used the string key END as well.  To be safe
    and unambiguous we return the string "__end__" explicitly when not
    reflecting, matching LangGraph's internal sentinel value precisely.

    Reflects when ALL of the following are true:
      1. REFLECT_ENABLED is True (operator toggle via .env).
      2. hallucination_risk > 0.7 (high risk threshold).
      3. reflect_loops < MAX_REFLECT_LOOPS (loop budget not exhausted).
    """
    cfg = get_settings()
    if not cfg.REFLECT_ENABLED:
        return END  # END == "__end__"; kept for semantic clarity

    h_risk = state.get("hallucination_risk", 0.0)
    loops  = state.get("reflect_loops", 0)

    if h_risk > 0.7 and loops < cfg.MAX_REFLECT_LOOPS:
        logger.info(
            "[%s] Routing to reflect (risk=%.2f, loops=%d/%d).",
            state.get("session_id"), h_risk, loops, cfg.MAX_REFLECT_LOOPS,
        )
        return "reflect"

    return END


def _after_reflect(state: GraphState) -> Literal["grade"]:
    """After Reflect: always re-grade the freshly retrieved docs."""
    return "grade"


# ─────────────────────────────────────────────
# GRAPH COMPILER
# ─────────────────────────────────────────────

def build_aura_graph(reranked_retriever: RerankedRetriever) -> Any:
    """
    Compile and return the AuraRAG LangGraph StateGraph.

    BUG-AG fix (v3.4): the graph is compiled ONCE at engine init time, not on
    every request.  Per-request invocation is pure graph dispatch overhead.

    The reranked_retriever is closed over in the retrieve and reflect node
    functions so both share the same pre-warmed retriever instance.
    """
    import functools

    retrieve_node = functools.partial(_node_retrieve, reranked_retriever=reranked_retriever)
    reflect_node  = functools.partial(_node_reflect,  reranked_retriever=reranked_retriever)

    workflow: StateGraph = StateGraph(GraphState)

    workflow.add_node("rewrite",  _node_rewrite)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade",    _node_grade)
    workflow.add_node("generate", _node_generate)
    workflow.add_node("reflect",  reflect_node)

    workflow.set_entry_point("rewrite")
    workflow.add_edge("rewrite",  "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("grade",    "generate")

    workflow.add_conditional_edges(
        "generate",
        _should_reflect,
        {"reflect": "reflect", END: END},
    )
    workflow.add_conditional_edges(
        "reflect",
        _after_reflect,
        {"grade": "grade"},
    )

    return workflow.compile()


# ─────────────────────────────────────────────
# RAG ENGINE
# ─────────────────────────────────────────────

class RAGEngine:
    """
    AuraRAG Enterprise Engine v3.4 — LangGraph Agentic Pipeline.

    Architecture:
      Ingest  → Chunk → Deduplicate → Embed → Store (ChromaDB + BM25)
      Query   → LangGraph (Rewrite → Retrieve → Grade → Generate → [Reflect])

    The LangGraph graph is compiled ONCE at __init__() and reused across
    requests (BUG-AG fix).  All graph nodes are async-first.

    All v3.3 fixes are fully preserved:
      BUG-Y  RerankedRetriever used everywhere (never bypassed).
      BUG-Z  asyncio.get_running_loop() in arerank().
      BUG-AB CrossEncoderReranker.shutdown() called from engine.shutdown().
      BUG-AC _evict_stale() reads get_settings().SESSION_TTL_MINUTES.
      BUG-AE SOURCE_SNIPPET_LEN read from settings, not hardcoded.
      BUG-S  top_k forwarded through the full call chain.
      BUG-U  SessionMemoryStore.clear() removes _last_access entry.
      BUG-V  Streaming exceptions propagated to SSE error frame.
      BUG-W  TextLoader uses UTF-8 + autodetect_encoding.
      BUG-X  _seen_hashes persisted in BM25 pickle; atomic pickle writes.

    New in v3.4:
      BUG-AF RerankedRetriever async path uses ainvoke() or to_thread().
      BUG-AG Graph compiled once at init; not rebuilt per request.
      BUG-AH Streaming uses LangGraph astream() — no bridge thread / race.
    """

    def __init__(self) -> None:
        logger.info("Initialising AuraRAG Engine v3.4...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
        )

        self._persist_dir = _resolve_persist_dir(settings.CHROMA_PERSIST_DIR)
        self.chroma_client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._bm25_pickle_path = self._persist_dir / "bm25.pkl"

        self._chroma = Chroma(
            client=self.chroma_client,
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=self.embeddings,
        )
        self.vector_store: Optional[Chroma] = (
            self._chroma if self._chroma._collection.count() > 0 else None
        )

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

        # BUG-AG fix: compile graph once at init.
        # If no documents are yet ingested the graph is None; it is compiled
        # (and recompiled after each ingest) via _rebuild_graph().
        self._graph: Any = None
        self._graph_top_k: Optional[int] = None
        if self.vector_store is not None and self._bm25_retriever is not None:
            self._rebuild_graph(top_k=5)

        logger.info("AuraRAG Engine v3.4 ready.")

    # ── GRAPH MANAGEMENT ──────────────────────────────────────────────────────

    def _rebuild_graph(self, top_k: int = 5) -> None:
        """
        (Re)compile the LangGraph with the current retriever state.

        Called after the first ingest and whenever top_k changes.
        For the common case of a single default top_k a single compiled graph
        is sufficient.  Heavy multi-top_k deployments can extend this with a
        small LRU cache of compiled graphs.
        """
        retriever   = self._build_reranked_retriever(top_k=top_k)
        self._graph = build_aura_graph(reranked_retriever=retriever)
        self._graph_top_k = top_k
        logger.info("LangGraph compiled (top_k=%d).", top_k)

    def _get_or_rebuild_graph(self, top_k: int) -> Any:
        """Return compiled graph, rebuilding only if top_k changed."""
        if self._graph is None or self._graph_top_k != top_k:
            self._rebuild_graph(top_k=top_k)
        return self._graph

    # ── BM25 PERSISTENCE ──────────────────────────────────────────────────────

    def _load_bm25_from_disk(self) -> None:
        if not self._bm25_pickle_path.exists():
            return
        try:
            with open(self._bm25_pickle_path, "rb") as fh:
                saved = pickle.load(fh)
            if isinstance(saved, dict):
                self._bm25_retriever = saved.get("retriever")
                self._all_docs       = saved.get("docs", [])
                # BUG-X fix (v3.2.0): restore _seen_hashes explicitly;
                # rebuild from docs for legacy pickles that pre-date this field.
                self._seen_hashes = saved.get(
                    "hashes",
                    {_content_hash(d) for d in self._all_docs},
                )
            else:
                # Legacy bare BM25Retriever from v3.0.0
                self._bm25_retriever = saved
                self._seen_hashes    = {_content_hash(d) for d in self._all_docs}
            logger.info("BM25 index restored (%d docs).", len(self._all_docs))
        except Exception as exc:
            logger.warning("Could not restore BM25 index: %s. Will rebuild on next ingest.", exc)

    def _save_bm25_to_disk(self) -> None:
        """
        Atomic write: pickle to .pkl.tmp then os.replace().
        BUG-X fix (v3.2.0): persists _seen_hashes alongside docs so that a
        restart does not reprocess already-ingested content.
        """
        try:
            self._bm25_pickle_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._bm25_pickle_path.with_suffix(".pkl.tmp")
            payload  = {
                "retriever": self._bm25_retriever,
                "docs":      self._all_docs,
                "hashes":    self._seen_hashes,
            }
            with open(tmp_path, "wb") as fh:
                pickle.dump(payload, fh)
            os.replace(tmp_path, self._bm25_pickle_path)  # atomic on POSIX
            logger.info("BM25 index saved (%d docs).", len(self._all_docs))
        except Exception as exc:
            logger.warning("Could not persist BM25 index: %s.", exc)

    # ── INGESTION ──────────────────────────────────────────────────────────────

    def ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
        logger.info("Ingesting %d raw document(s)...", len(documents))
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
            logger.info("Skipped %d duplicate chunk(s).", dup_count)

        if not new_chunks:
            return {"chunks_ingested": 0, "duplicates_skipped": dup_count, "status": "success"}

        logger.info("Adding %d new chunks...", len(new_chunks))

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
        self._bm25_retriever   = BM25Retriever.from_documents(self._all_docs)
        self._bm25_retriever.k = _MAX_RETRIEVAL_DOCS
        self._save_bm25_to_disk()

        # Recompile the graph with the updated retriever.
        self._rebuild_graph(top_k=5)

        result = {
            "chunks_ingested":    len(new_chunks),
            "duplicates_skipped": dup_count,
            "status":             "success",
        }
        logger.info("Ingestion complete: %s", result)
        return result

    # ── RETRIEVAL BUILDERS ────────────────────────────────────────────────────

    def _build_hybrid_retriever(self) -> EnsembleRetriever:
        if self.vector_store is None or self._bm25_retriever is None:
            raise RuntimeError("No documents ingested. Upload files first.")
        dense = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": _MAX_RETRIEVAL_DOCS, "fetch_k": _MAX_RETRIEVAL_DOCS * 2},
        )
        return EnsembleRetriever(
            retrievers=[dense, self._bm25_retriever],
            weights=[0.6, 0.4],
        )

    def _build_reranked_retriever(self, top_k: int = 5) -> RerankedRetriever:
        """
        BUG-Y fix (v3.3): wraps hybrid search + cross-encoder reranking in one
        BaseRetriever-compatible object so the reranker is never bypassed.
        """
        return RerankedRetriever(
            hybrid_retriever=self._build_hybrid_retriever(),
            cross_encoder=self.reranker,
            top_k=top_k,
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Standalone hybrid search + cross-encoder re-rank (no LLM)."""
        hybrid     = self._build_hybrid_retriever()
        candidates = hybrid.invoke(query)
        return self.reranker.rerank(query, candidates, top_k=top_k)

    # ── STATE BUILDER ─────────────────────────────────────────────────────────

    def _build_initial_state(
        self,
        question: str,
        provider: str,
        model: str,
        api_key: str,
        session_id: str,
        top_k: int,
        system_prompts: Optional[Dict[str, str]] = None,
    ) -> GraphState:
        """Assemble the initial GraphState before invoking the graph."""
        return GraphState(
            question=question,
            provider=provider,
            model=model,
            api_key=api_key,
            session_id=session_id,
            top_k=top_k,
            system_prompts=system_prompts or {},
            chat_history_text=self._session_store.format_history(session_id),
            rewritten_query="",
            reflection_feedback="",
            retrieved_docs=[],
            relevant_docs=[],
            graded_count=0,
            answer="",
            hallucination_risk=0.0,
            reflect_loops=0,
            needs_revision=False,
            pipeline_trace=[],
        )

    def _finalise_response(self, final_state: GraphState, session_id: str) -> Dict[str, Any]:
        """
        Extract the public-facing response dict from the final graph state and
        persist the completed Q/A turn into session memory.
        """
        answer = (
            final_state.get("answer")
            or "I do not have enough information in the provided documents to answer this question."
        )
        self._session_store.save_turn(session_id, final_state["question"], answer)

        snippet_len  = get_settings().SOURCE_SNIPPET_LEN
        top_k        = final_state.get("top_k", 5)
        context_docs = _dedupe_documents(
            final_state.get("relevant_docs") or final_state.get("retrieved_docs", [])
        )[:top_k]

        return {
            "answer":         answer,
            "sources":        [
                {"content": doc.page_content[:snippet_len], "metadata": doc.metadata}
                for doc in context_docs
            ],
            "chat_history":   [
                m.content
                for m in self._session_store.get(session_id).chat_memory.messages
            ],
            "session_id":     session_id,
            "pipeline_trace": final_state.get("pipeline_trace", []),
            "graded_chunks":  final_state.get("graded_count", 0),
            "reflect_loops":  final_state.get("reflect_loops", 0),
        }

    # ── ASYNCHRONOUS QUERY ─────────────────────────────────────────────────────

    async def aquery(
        self,
        question:   str,
        provider:   str,
        model:      str,
        api_key:    str,
        session_id: str = "default",
        top_k:      int = 5,
        system_prompts: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Primary async query entrypoint.
        Runs the full LangGraph pipeline: Rewrite → Retrieve → Grade → Generate → [Reflect].
        """
        if self.vector_store is None:
            return {
                "answer":         "No documents ingested yet. Upload files first.",
                "sources":        [],
                "chat_history":   [],
                "session_id":     session_id,
                "pipeline_trace": [],
                "graded_chunks":  0,
                "reflect_loops":  0,
            }

        graph         = self._get_or_rebuild_graph(top_k=top_k)
        initial_state = self._build_initial_state(question, provider, model, api_key, session_id, top_k, system_prompts=system_prompts)

        logger.info(
            "[%s] aquery via %s/%s (top_k=%d): %r",
            session_id, provider, model, top_k, question[:80],
        )

        final_state: GraphState = await graph.ainvoke(initial_state)
        return self._finalise_response(final_state, session_id)

    # ── SYNCHRONOUS QUERY (thin wrapper) ──────────────────────────────────────

    def query(
        self,
        question:   str,
        provider:   str,
        model:      str,
        api_key:    str,
        session_id: str = "default",
        top_k:      int = 5,
        system_prompts: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous entrypoint retained for backward-compatibility with tests.

        The FastAPI router calls this via asyncio.to_thread(), so it executes
        in a worker thread where there is no running event loop — making
        asyncio.run() safe here.

        BUG-AO fix (v3.5): the previous implementation raised RuntimeError and
        then caught it in the same except block, checking "event loop" in the
        message — which matched the error it just raised, causing a confusing
        re-raise path instead of the intended guard.  The fix uses
        asyncio.get_running_loop() which raises RuntimeError("no running event
        loop") only when there is NO loop, and succeeds silently when one IS
        running — so the guard is simply: if it succeeds, raise our advisory
        error; if it raises, we're in the right thread context.

        Note: if you are already inside a running event loop (e.g. in a Jupyter
        notebook), call aquery() directly instead.
        """
        # BUG-AO fix: get_running_loop() raises RuntimeError when there is no
        # running loop (correct context for asyncio.run()); it returns the loop
        # object when we ARE inside one (wrong context — raise advisory error).
        try:
            asyncio.get_running_loop()
            # If we reach here a loop IS running — wrong context for this method.
            raise RuntimeError(
                "engine.query() was called from inside a running event loop. "
                "Use engine.aquery() instead."
            )
        except RuntimeError as exc:
            # Only swallow the "no running event loop" error from get_running_loop().
            # Re-raise our advisory error and any other RuntimeError.
            if "no running event loop" not in str(exc).lower() and "no current event loop" not in str(exc).lower():
                raise
        return asyncio.run(
            self.aquery(question, provider, model, api_key, session_id=session_id, top_k=top_k)
        )

    # ── TRUE SSE STREAMING ─────────────────────────────────────────────────────

    async def stream_query(
        self,
        question:   str,
        provider:   str,
        model:      str,
        api_key:    str,
        session_id: str = "default",
        top_k:      int = 5,
        system_prompts: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that streams tokens from the Generate node.

        BUG-AH fix (v3.4): v3.3 bridged an asyncio event loop and a worker
        thread via AsyncIteratorCallbackHandler + asyncio.to_thread(), creating
        a data race on the internal asyncio.Queue at shutdown.  The new
        implementation uses LangGraph's native astream() API — fully async-first,
        no bridge thread, no race condition.

        BUG-V fix (v3.2.0) preserved: exceptions are caught and re-raised so
        the SSE layer emits an error frame rather than silently hanging.

        Streaming strategy:
          - Pre-generation nodes (rewrite, retrieve, grade) run to completion —
            their latency is sub-second and not worth streaming.
          - The Generate node LLM call is streamed token-by-token via the
            "messages" stream mode, filtered to the "generate" LangGraph node.
          - After streaming completes the turn is saved to session memory.
        """
        if self.vector_store is None:
            yield "No documents ingested yet. Upload files first."
            return

        graph         = self._get_or_rebuild_graph(top_k=top_k)
        initial_state = self._build_initial_state(question, provider, model, api_key, session_id, top_k, system_prompts=system_prompts)

        logger.info(
            "[%s] stream_query via %s/%s (top_k=%d): %r",
            session_id, provider, model, top_k, question[:80],
        )

        full_answer = ""
        final_state: Optional[GraphState] = None

        try:
            # stream_mode=["messages", "updates"] gives us both token events
            # from the LLM and node completion events for state capture.
            # BUG-AI fix: use the locally-retrieved `graph` (which respects
            # top_k changes via _get_or_rebuild_graph) instead of self._graph.
            async for chunk in graph.astream(
                initial_state,
                stream_mode=["messages", "updates"],
            ):
                # BUG-AJ fix: when stream_mode is a list, LangGraph yields
                # tuples of (mode_str, data) — NOT dicts with a "type" key.
                # The old code called chunk.get("type") on a tuple, which
                # always returned None and silently skipped every event.
                if not isinstance(chunk, tuple) or len(chunk) != 2:
                    continue
                chunk_type, chunk_data = chunk

                # Capture the final state from "updates" events.
                if chunk_type == "updates":
                    if isinstance(chunk_data, dict):
                        # Each "updates" payload is {node_name: node_output_dict}.
                        # Merge all node outputs into final_state — last write wins.
                        for node_output in chunk_data.values():
                            if isinstance(node_output, dict):
                                final_state = {**(final_state or {}), **node_output}  # type: ignore[misc]

                # Stream tokens from the Generate node's LLM call.
                elif chunk_type == "messages":
                    message, metadata = chunk_data
                    if metadata.get("langgraph_node") != "generate":
                        continue
                    token = getattr(message, "content", "")
                    if isinstance(token, list):
                        token = "".join(
                            part.get("text", "") for part in token
                            if isinstance(part, dict)
                        )
                    token = str(token)
                    if token:
                        full_answer += token
                        yield token

        except Exception as exc:
            logger.exception("[%s] stream_query error", session_id)
            raise exc

        # If no tokens streamed (non-streaming provider or Ollama), fall back
        # to reading the answer from the final state captured via "updates".
        if not full_answer and final_state:
            fallback = final_state.get("answer", "")
            if fallback:
                yield fallback
                full_answer = fallback

        # Persist completed turn regardless of how the answer arrived.
        if full_answer:
            clean, _ = _extract_hallucination_risk(full_answer)
            self._session_store.save_turn(session_id, question, clean)
            logger.info("[%s] Streaming complete (%d chars).", session_id, len(full_answer))

    # ── UTILITIES ──────────────────────────────────────────────────────────────

    def clear_memory(self, session_id: str = "default") -> None:
        self._session_store.clear(session_id)

    def clear_all_memory(self) -> None:
        self._session_store.clear_all()

    def shutdown(self) -> None:
        """
        BUG-AB fix (v3.3): release all engine resources during application teardown.
        Called from the FastAPI lifespan cleanup block.
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
            "bm25_docs":       str(bm25_count),  # BUG-AD fix (v3.3)
            "active_sessions": str(self._session_store.active_sessions),
        }
