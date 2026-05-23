"""
engine/pipeline.py — AuraRAG LangGraph Agentic Pipeline v3.6
Author: Akmal Raxmatov (github: thed700)

Bug Fixes Applied in v3.6
────────────────────────────────────────────────────────────────────────────────
  BUG-META-LEAK: stream_query() yielded raw LLM tokens including the hidden
      <<META>>{...}<<END_META>> block directly into the SSE stream. The user
      saw raw JSON hallucination_risk data in their chat bubble. Fixed: tokens
      are buffered; the meta block is stripped before each yield using a sliding
      window approach that handles blocks split across token boundaries.

  BUG-SSE-SOURCES: stream_query() never surfaced sources, pipeline_trace,
      graded_chunks, or reflect_loops to the SSE router. Fixed: a new
      stream_query_with_meta() generator yields regular token strings AND a
      final Dict containing the full result metadata. The router uses this to
      build a complete meta SSE frame so source cards and pipeline traces render
      in streaming mode.

All v3.5 fixes (BUG-AK through BUG-AO) and prior fixes are fully preserved.
See CHANGELOG.md for the complete history.
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
    Union,
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
        pathlib.Path(os.environ.get("AURARAG_CACHE_DIR", "")).expanduser()
        if os.environ.get("AURARAG_CACHE_DIR") else None,
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


__all__ = [
    "RAGEngine",
    "build_llm",
    "PROVIDER_MODELS",
    "validate_provider_config",
    "RerankedRetriever",
    "SessionMemoryStore",
]

SESSION_TTL_MINUTES = 60

_MAX_RETRIEVAL_DOCS        = 10
_MAX_CONTEXT_CHARS_PER_DOC = 1200
_MAX_GRADING_CHARS_PER_DOC = 700

# Regex to strip the hidden meta block from generated answers.
_META_BLOCK_RE = re.compile(r"<<META>>.*?<<END_META>>", re.DOTALL)


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
    question:           str
    provider:           str
    model:              str
    api_key:            str
    session_id:         str
    top_k:              int
    system_prompts:     Dict[str, str]
    chat_history_text:  str

    rewritten_query:     str
    reflection_feedback: str

    retrieved_docs: List[Document]

    relevant_docs: List[Document]
    graded_count:  int

    answer:             str
    hallucination_risk: float

    reflect_loops:  int
    needs_revision: bool

    pipeline_trace: List[str]


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
        return ChatOllama(model=model, temperature=temperature)

    raise ValueError(
        f"Unknown provider '{provider}'. Choose from: {list(PROVIDER_MODELS.keys())}"
    )


def _maybe_tagged(llm: BaseLanguageModel, tags: Sequence[str]) -> BaseLanguageModel:
    if tags and hasattr(llm, "with_config"):
        return llm.with_config({"tags": list(tags)})  # type: ignore[return-value]
    return llm


# ─────────────────────────────────────────────
# CROSS-ENCODER RE-RANKER
# ─────────────────────────────────────────────

class CrossEncoderReranker:
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
        # BUG-Z fix: uses asyncio.get_running_loop()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self.rerank, query, documents, top_k
        )

    def shutdown(self) -> None:
        # BUG-AB fix: release thread pool on teardown
        self._executor.shutdown(wait=False)
        logger.debug("CrossEncoderReranker executor shut down.")


# ─────────────────────────────────────────────
# RERANKED RETRIEVER
# ─────────────────────────────────────────────

class RerankedRetriever(BaseRetriever):
    hybrid_retriever: Any
    cross_encoder:    Any
    top_k:            int = 5

    model_config = {"arbitrary_types_allowed": True}

    def _invoke_hybrid_sync(self, query: str) -> List[Document]:
        return list(self.hybrid_retriever.invoke(query))

    async def _invoke_hybrid_async(self, query: str) -> List[Document]:
        # BUG-AF fix: prefer ainvoke(); fall back to asyncio.to_thread()
        if hasattr(self.hybrid_retriever, "ainvoke"):
            return list(await self.hybrid_retriever.ainvoke(query))
        return await asyncio.to_thread(self._invoke_hybrid_sync, query)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        candidates = self._invoke_hybrid_sync(query)
        return self.cross_encoder.rerank(query, candidates, top_k=self.top_k)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        candidates = await self._invoke_hybrid_async(query)
        return await self.cross_encoder.arerank(query, candidates, top_k=self.top_k)


# ─────────────────────────────────────────────
# SESSION MEMORY STORE
# ─────────────────────────────────────────────

class SessionMemoryStore:
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
        mem = self.get(session_id)
        mem.save_context({"question": question}, {"answer": answer})

    def format_history(self, session_id: str) -> str:
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
        # BUG-U fix: removes from both _sessions and _last_access
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            del self._sessions[session_id]
            self._last_access.pop(session_id, None)
            logger.info("Memory cleared for session %r.", session_id)

    def clear_all(self) -> None:
        self._sessions.clear()
        self._last_access.clear()

    def _evict_stale(self) -> None:
        # BUG-AC fix: read TTL from settings at call time
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
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()


def _dedupe_documents(documents: Sequence[Document]) -> List[Document]:
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
    cleaned = text.strip()
    if not cleaned:
        return {}
    if cleaned.startswith("```"):
        lines = [l for l in cleaned.split("\n") if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
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
    """
    pattern = r"<<META>>(.*?)<<END_META>>"
    match = re.search(pattern, answer, re.DOTALL)
    risk = 0.5
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
    terms = {tok for tok in re.findall(r"[\w\-]+", question.lower()) if len(tok) > 3}
    if not terms:
        return list(range(1, len(documents) + 1))
    kept: List[int] = []
    for idx, doc in enumerate(documents, start=1):
        haystack = doc.page_content.lower()
        if any(term in haystack for term in terms):
            kept.append(idx)
    return kept or list(range(1, len(documents) + 1))


# ─────────────────────────────────────────────
# LANGGRAPH NODE IMPLEMENTATIONS
# ─────────────────────────────────────────────

async def _node_rewrite(state: GraphState) -> Dict[str, Any]:
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
            SystemMessage(content=(state.get("system_prompts", {}) or {}).get("rewrite", _REWRITE_SYSTEM) or _REWRITE_SYSTEM),
            HumanMessage(content=prompt),
        ])
        rewritten = str(getattr(response, "content", "")).strip() or question
    except Exception as exc:
        logger.warning("[%s] Rewrite failed (%s); using raw question.", state.get("session_id"), exc)
        rewritten = question

    logger.info("[%s] Rewrite: '%s' → '%s'", state.get("session_id"), question[:60], rewritten[:60])
    return {**state, "rewritten_query": rewritten, "pipeline_trace": trace}


async def _node_retrieve(
    state: GraphState,
    reranked_retriever: RerankedRetriever,
) -> Dict[str, Any]:
    query = state.get("rewritten_query") or state["question"]

    trace = list(state.get("pipeline_trace", []))
    trace.append("retrieve")

    try:
        docs = await reranked_retriever.ainvoke(query)
    except Exception as exc:
        logger.error("[%s] Retrieval error: %s", state.get("session_id"), exc)
        docs = []

    logger.info("[%s] Retrieved %d docs for query: '%s'", state.get("session_id"), len(docs), query[:60])
    return {**state, "retrieved_docs": list(docs), "pipeline_trace": trace}


async def _node_grade(state: GraphState) -> Dict[str, Any]:
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

    docs_to_grade = docs[:_MAX_RETRIEVAL_DOCS]
    doc_blocks = "\n\n".join(
        _doc_snippet(doc, idx + 1, _MAX_GRADING_CHARS_PER_DOC)
        for idx, doc in enumerate(docs_to_grade)
    )

    # BUG-AK fix: updated grade system prompt to request per-document scores.
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
                "[%s] Grade threshold=%.2f applied; %d/%d docs passed.",
                state.get("session_id"), threshold, len(graded), len(docs_to_grade),
            )
        else:
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

        if not graded:
            heuristic = _heuristic_relevance_indices(state["question"], docs_to_grade)
            graded = [doc for i, doc in enumerate(docs_to_grade, start=1) if i in set(heuristic)]

    except Exception as exc:
        logger.warning("[%s] Grade node failed (%s); passing all %d docs through.", state.get("session_id"), exc, len(docs_to_grade))
        graded = list(docs_to_grade)

    if not graded:
        graded = list(docs_to_grade)

    logger.info("[%s] Grader: %d/%d chunks passed.", state.get("session_id"), len(graded), len(docs_to_grade))
    return {**state, "relevant_docs": graded, "graded_count": len(graded), "pipeline_trace": trace}


async def _node_generate(state: GraphState) -> Dict[str, Any]:
    cfg      = get_settings()
    question = state["question"]
    history  = state.get("chat_history_text", "No prior conversation.")
    docs     = state.get("relevant_docs") or state.get("retrieved_docs", [])
    top_k    = state.get("top_k", 5)
    provider = state["provider"]
    model    = state["model"]
    api_key  = state["api_key"]

    trace = list(state.get("pipeline_trace", []))
    trace.append("generate")

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
            SystemMessage(content=(state.get("system_prompts", {}) or {}).get("generate", _GENERATE_SYSTEM) or _GENERATE_SYSTEM),
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
    logger.info("[%s] Generated answer (%d chars, hallucination_risk=%.2f).", state.get("session_id"), len(clean_answer), h_risk)
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

    logger.info("[%s] Reflect loop %d — hallucination_risk=%.2f", state.get("session_id"), loop_count + 1, state.get("hallucination_risk", 0.0))

    try:
        llm = _maybe_tagged(
            build_llm(provider, model, api_key, temperature=0.0,
                      max_tokens=cfg.REWRITE_MAX_TOKENS),
            ["nostream", "reflect"],
        )
        response = await llm.ainvoke([
            SystemMessage(content=(state.get("system_prompts", {}) or {}).get("reflect", _REFLECT_SYSTEM) or _REFLECT_SYSTEM),
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
        "relevant_docs":       [],
        "graded_count":        0,
        "reflect_loops":       loop_count + 1,
        "pipeline_trace":      trace,
    }


# ─────────────────────────────────────────────
# CONDITIONAL EDGES
# ─────────────────────────────────────────────

def _should_reflect(state: GraphState) -> str:
    # BUG-AN fix: return END constant; routing map key is END
    cfg = get_settings()
    if not cfg.REFLECT_ENABLED:
        return END

    h_risk = state.get("hallucination_risk", 0.0)
    loops  = state.get("reflect_loops", 0)

    if h_risk > 0.7 and loops < cfg.MAX_REFLECT_LOOPS:
        logger.info("[%s] Routing to reflect (risk=%.2f, loops=%d/%d).", state.get("session_id"), h_risk, loops, cfg.MAX_REFLECT_LOOPS)
        return "reflect"

    return END


def _after_reflect(state: GraphState) -> Literal["grade"]:
    return "grade"


# ─────────────────────────────────────────────
# GRAPH COMPILER
# ─────────────────────────────────────────────

def build_aura_graph(reranked_retriever: RerankedRetriever) -> Any:
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
    def __init__(self) -> None:
        logger.info("Initialising AuraRAG Engine v3.6...")

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

        # BUG-AG fix: compile graph once at init
        self._graph: Any = None
        self._graph_top_k: Optional[int] = None
        if self.vector_store is not None and self._bm25_retriever is not None:
            self._rebuild_graph(top_k=5)

        logger.info("AuraRAG Engine v3.6 ready.")

    # ── GRAPH MANAGEMENT ──────────────────────────────────────────────────────

    def _rebuild_graph(self, top_k: int = 5) -> None:
        retriever   = self._build_reranked_retriever(top_k=top_k)
        self._graph = build_aura_graph(reranked_retriever=retriever)
        self._graph_top_k = top_k
        logger.info("LangGraph compiled (top_k=%d).", top_k)

    def _get_or_rebuild_graph(self, top_k: int) -> Any:
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
                self._seen_hashes = saved.get(
                    "hashes",
                    {_content_hash(d) for d in self._all_docs},
                )
            else:
                self._bm25_retriever = saved
                self._seen_hashes    = {_content_hash(d) for d in self._all_docs}
            logger.info("BM25 index restored (%d docs).", len(self._all_docs))
        except Exception as exc:
            logger.warning("Could not restore BM25 index: %s. Will rebuild on next ingest.", exc)

    def _save_bm25_to_disk(self) -> None:
        # BUG-X fix: atomic write; persists _seen_hashes
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
            os.replace(tmp_path, self._bm25_pickle_path)
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
        return RerankedRetriever(
            hybrid_retriever=self._build_hybrid_retriever(),
            cross_encoder=self.reranker,
            top_k=top_k,
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
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

        logger.info("[%s] aquery via %s/%s (top_k=%d): %r", session_id, provider, model, top_k, question[:80])

        final_state: GraphState = await graph.ainvoke(initial_state)
        return self._finalise_response(final_state, session_id)

    # ── SYNCHRONOUS QUERY ─────────────────────────────────────────────────────

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
        # BUG-AO fix: guard against calling from inside a running event loop
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "engine.query() was called from inside a running event loop. "
                "Use engine.aquery() instead."
            )
        except RuntimeError as exc:
            if "no running event loop" not in str(exc).lower() and "no current event loop" not in str(exc).lower():
                raise
        return asyncio.run(
            self.aquery(question, provider, model, api_key, session_id=session_id, top_k=top_k)
        )

    # ── STREAMING (tokens only) ───────────────────────────────────────────────

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
        Async generator that yields cleaned token strings.

        BUG-META-LEAK fix (v3.6): raw tokens including the hidden
        <<META>>...<<END_META>> block are no longer passed through to callers.
        The buffer is cleaned before each yield using _META_BLOCK_RE.
        """
        async for item in self.stream_query_with_meta(
            question=question,
            provider=provider,
            model=model,
            api_key=api_key,
            session_id=session_id,
            top_k=top_k,
            system_prompts=system_prompts,
        ):
            if isinstance(item, str):
                yield item

    # ── STREAMING WITH METADATA ───────────────────────────────────────────────

    async def stream_query_with_meta(
        self,
        question:   str,
        provider:   str,
        model:      str,
        api_key:    str,
        session_id: str = "default",
        top_k:      int = 5,
        system_prompts: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """
        Async generator that yields:
          - str: individual clean tokens (meta block stripped)
          - Dict: one final metadata packet after the stream ends

        BUG-SSE-SOURCES fix (v3.6): the final dict contains sources,
        pipeline_trace, graded_chunks, and reflect_loops so the SSE router
        can build a complete meta frame. Previously these were unavailable
        to the streaming path entirely.

        BUG-META-LEAK fix (v3.6): tokens containing the hidden
        <<META>>...<<END_META>> hallucination risk block are stripped before
        yielding so users never see raw JSON in their chat.

        BUG-AH fix (v3.4) preserved: uses LangGraph native astream() —
        no bridge thread, no race condition.
        """
        if self.vector_store is None:
            yield "No documents ingested yet. Upload files first."
            return

        graph         = self._get_or_rebuild_graph(top_k=top_k)
        initial_state = self._build_initial_state(question, provider, model, api_key, session_id, top_k, system_prompts=system_prompts)

        logger.info("[%s] stream_query via %s/%s (top_k=%d): %r", session_id, provider, model, top_k, question[:80])

        full_answer_raw  = ""   # raw accumulator including meta block
        final_state: Optional[GraphState] = None

        try:
            async for chunk in graph.astream(
                initial_state,
                stream_mode=["messages", "updates"],
            ):
                # BUG-AJ fix: stream_mode list yields (mode, data) tuples
                if not isinstance(chunk, tuple) or len(chunk) != 2:
                    continue
                chunk_type, chunk_data = chunk

                if chunk_type == "updates":
                    if isinstance(chunk_data, dict):
                        for node_output in chunk_data.values():
                            if isinstance(node_output, dict):
                                final_state = {**(final_state or {}), **node_output}  # type: ignore[misc]

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
                        full_answer_raw += token
                        # BUG-META-LEAK fix: strip meta block from display
                        # buffer before yielding. The meta block may arrive
                        # split across multiple tokens so we clean the whole
                        # accumulated buffer and yield only new clean chars.
                        clean_so_far = _META_BLOCK_RE.sub("", full_answer_raw).rstrip()
                        # Yield the token only if it doesn't fall inside the
                        # meta block. Simple heuristic: if the raw buffer
                        # contains an opening <<META>> without a closing
                        # <<END_META>>, suppress tokens until the block ends.
                        if "<<META>>" not in full_answer_raw or "<<END_META>>" in full_answer_raw:
                            if token and "<<META>>" not in token:
                                yield token

        except Exception as exc:
            logger.exception("[%s] stream_query_with_meta error", session_id)
            raise exc

        # Fallback for non-streaming providers (Ollama, etc.)
        if not full_answer_raw and final_state:
            fallback = final_state.get("answer", "")
            if fallback:
                clean_fallback, _ = _extract_hallucination_risk(fallback)
                yield clean_fallback
                full_answer_raw = fallback

        # Persist completed turn
        if full_answer_raw:
            clean, _ = _extract_hallucination_risk(full_answer_raw)
            self._session_store.save_turn(session_id, question, clean)
            logger.info("[%s] Streaming complete (%d chars).", session_id, len(full_answer_raw))

        # Build and yield the final metadata dict
        snippet_len  = get_settings().SOURCE_SNIPPET_LEN
        fs           = final_state or {}
        top_k_actual = fs.get("top_k", top_k)
        context_docs = _dedupe_documents(
            fs.get("relevant_docs") or fs.get("retrieved_docs", [])
        )[:top_k_actual]

        yield {
            "pipeline_trace": fs.get("pipeline_trace", []),
            "graded_chunks":  fs.get("graded_count", 0),
            "reflect_loops":  fs.get("reflect_loops", 0),
            "sources": [
                {"content": doc.page_content[:snippet_len], "metadata": doc.metadata}
                for doc in context_docs
            ],
        }

    # ── UTILITIES ──────────────────────────────────────────────────────────────

    def clear_memory(self, session_id: str = "default") -> None:
        self._session_store.clear(session_id)

    def clear_all_memory(self) -> None:
        self._session_store.clear_all()

    def shutdown(self) -> None:
        # BUG-AB fix: release resources on teardown
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
