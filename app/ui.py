"""
ui.py — AuraRAG v4.0 Chat Interface
Author: Akmal Raxmatov (github: thed700)

Changes v4.0:
  FEAT-UI:  Complete redesign — ChatGPT/Claude-style layout with avatar
            bubbles, markdown+code rendering, copy-to-clipboard via JS bridge.
  FEAT-PROG: Live LangGraph step progress component (Rewrite → Retrieve →
             Grade → Generate) shown while the pipeline runs.
  FEAT-FILES: Sidebar file uploader now accepts PDF, TXT, CSV, JSON, XLSX,
              Parquet — matching the expanded /ingest backend.
  FEAT-OLLAMA: Custom text input for Ollama model names (any local model).
  FEAT-TOPK:  top_k slider exposed in sidebar Advanced settings expander.
  FEAT-SYSP:  System-prompt override textarea in Advanced settings.
  FIX-AA (carried): top_k included in both /query and /query/stream payloads.
  FIX-P  (carried): session_id per browser tab via st.runtime session.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Generator

import requests
import streamlit as st

from app.constants import (
    PROVIDER_MODELS as FALLBACK_PROVIDERS,
    friendly_model_label,
    validate_provider_config,
)

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AuraRAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

# ─────────────────────────────────────────────
# DESIGN SYSTEM  — Obsidian + Amber
# ─────────────────────────────────────────────

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Lora:ital,wght@0,400;0,500;1,400&display=swap');

:root {
    --bg:           #09090b;
    --surface:      #111113;
    --surface2:     #18181b;
    --surface3:     #1f1f23;
    --border:       #27272a;
    --border2:      #3f3f46;
    --amber:        #f59e0b;
    --amber-dim:    #b45309;
    --amber-glow:   rgba(245,158,11,.10);
    --amber-glow2:  rgba(245,158,11,.18);
    --text:         #fafafa;
    --text2:        #a1a1aa;
    --text3:        #71717a;
    --green:        #22c55e;
    --red:          #ef4444;
    --blue:         #60a5fa;
    --radius:       12px;
    --radius-sm:    8px;
    --font-head:    'Syne', sans-serif;
    --font-body:    'Lora', serif;
    --font-mono:    'JetBrains Mono', monospace;
}

/* ── Base reset ────────────────────────────── */
html, body, [class*="css"] {
    font-family: var(--font-body);
    color: var(--text);
    background: var(--bg);
}
.stApp { background: var(--bg) !important; }
.block-container { padding: 0 1.5rem 4rem !important; max-width: 860px !important; }

/* ── Sidebar ────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }

[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput  > div > div > input,
[data-testid="stSidebar"] .stTextArea   > div > div > textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    font-size: .8rem !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div:focus-within,
[data-testid="stSidebar"] .stTextInput  > div > div > input:focus,
[data-testid="stSidebar"] .stTextArea   > div > div > textarea:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 2px var(--amber-glow) !important;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text2);
    font-family: var(--font-head);
    font-size: .78rem;
    font-weight: 500;
    letter-spacing: .04em;
    transition: all .16s;
    padding: .45rem 1rem;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--amber);
    color: var(--amber);
    background: var(--amber-glow);
}

/* ── Sidebar logo area ──────────────────────── */
.sb-header {
    padding: 1.4rem 1.2rem 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: .5rem;
}
.sb-logo {
    font-family: var(--font-head);
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--amber);
    letter-spacing: -.03em;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.sb-tag {
    font-family: var(--font-mono);
    font-size: .6rem;
    color: var(--text3);
    text-transform: uppercase;
    letter-spacing: .1em;
    margin-top: 3px;
}
.sb-section { padding: .6rem 1.2rem; }
.sb-label {
    font-family: var(--font-head);
    font-size: .68rem;
    font-weight: 600;
    color: var(--text3);
    text-transform: uppercase;
    letter-spacing: .1em;
    margin-bottom: .5rem;
}

/* ── Status pills ──────────────────────────── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: .7rem;
    font-family: var(--font-mono);
    font-weight: 500;
    margin: 2px 2px 2px 0;
}
.pill-ok   { background: rgba(34,197,94,.1);  color: #22c55e; border: 1px solid rgba(34,197,94,.22); }
.pill-err  { background: rgba(239,68,68,.1);  color: #ef4444; border: 1px solid rgba(239,68,68,.22); }
.pill-warn { background: rgba(245,158,11,.1); color: #f59e0b; border: 1px solid rgba(245,158,11,.22); }

/* ── Chat header ───────────────────────────── */
.chat-header {
    padding: 1.4rem 0 .6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}
.chat-title {
    font-family: var(--font-head);
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -.04em;
}
.chat-subtitle {
    font-family: var(--font-mono);
    font-size: .7rem;
    color: var(--text3);
    margin-top: 4px;
    display: flex;
    gap: .8rem;
    flex-wrap: wrap;
    align-items: center;
}

/* ── Model badge ───────────────────────────── */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--amber-glow);
    border: 1px solid var(--amber-dim);
    border-radius: 6px;
    padding: 2px 8px;
    font-family: var(--font-mono);
    font-size: .68rem;
    color: var(--amber);
}
.stream-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(34,197,94,.08);
    border: 1px solid rgba(34,197,94,.22);
    border-radius: 6px;
    padding: 2px 8px;
    font-family: var(--font-mono);
    font-size: .68rem;
    color: #22c55e;
}

/* ── Chat message bubbles ──────────────────── */
[data-testid="stChatMessage"] {
    border-radius: var(--radius) !important;
    padding: .85rem 1rem !important;
    margin-bottom: .55rem !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
}
/* User messages slightly different */
[data-testid="stChatMessage"][data-testid*="user"] {
    background: var(--surface2) !important;
    border-color: var(--border2) !important;
}

/* Force avatar area styles */
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] > div {
    background: var(--amber-glow2) !important;
    border: 1px solid var(--amber-dim) !important;
    color: var(--amber) !important;
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
}
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] > div {
    background: var(--surface3) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text2) !important;
}

/* Message text */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    font-family: var(--font-body) !important;
    font-size: .92rem !important;
    line-height: 1.65 !important;
    color: var(--text) !important;
}

/* Code blocks inside messages */
[data-testid="stChatMessage"] code,
[data-testid="stChatMessage"] pre {
    font-family: var(--font-mono) !important;
    font-size: .8rem !important;
    background: var(--surface3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stChatMessage"] pre { padding: .8rem 1rem !important; }

/* ── LangGraph progress stepper ─────────────── */
.lg-progress {
    display: flex;
    align-items: center;
    gap: 0;
    margin: .6rem 0 .9rem;
    font-family: var(--font-mono);
    font-size: .7rem;
}
.lg-step {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 20px;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text3);
    transition: all .25s;
    white-space: nowrap;
}
.lg-step.active {
    background: var(--amber-glow2);
    border-color: var(--amber);
    color: var(--amber);
    font-weight: 600;
}
.lg-step.done {
    background: rgba(34,197,94,.08);
    border-color: rgba(34,197,94,.3);
    color: #22c55e;
}
.lg-arrow {
    color: var(--border2);
    margin: 0 3px;
    flex-shrink: 0;
}

/* ── Copy button overlay ───────────────────── */
.copy-wrap { position: relative; }
.copy-btn {
    position: absolute;
    top: 6px;
    right: 8px;
    background: var(--surface3);
    border: 1px solid var(--border2);
    border-radius: 5px;
    padding: 2px 8px;
    font-family: var(--font-mono);
    font-size: .65rem;
    color: var(--text2);
    cursor: pointer;
    transition: all .15s;
    z-index: 10;
}
.copy-btn:hover { border-color: var(--amber); color: var(--amber); }

/* ── Citation chips ────────────────────────── */
.cite-row { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; }
.cite-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 9px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: .68rem;
    color: var(--text3);
}

/* ── Source card ───────────────────────────── */
.src-card {
    background: var(--surface2);
    border-left: 3px solid var(--amber);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: .8rem;
}
.src-meta {
    font-family: var(--font-mono);
    font-size: .66rem;
    color: var(--amber);
    margin-bottom: 5px;
    display: flex;
    gap: 12px;
}
.src-snippet { color: var(--text2); font-family: var(--font-body); line-height: 1.5; }

/* ── Empty state ───────────────────────────── */
.empty-state {
    text-align: center;
    padding: 3.5rem 1rem 1.5rem;
    color: var(--text3);
}
.empty-icon { font-size: 2.6rem; margin-bottom: .6rem; }
.empty-title {
    font-family: var(--font-head);
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text2);
    margin-bottom: .35rem;
    letter-spacing: -.02em;
}
.empty-sub { font-size: .82rem; color: var(--text3); }

/* ── Starter prompt buttons ────────────────── */
[data-testid="stButton"] > button[kind="secondary"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    color: var(--text2) !important;
    font-family: var(--font-body) !important;
    font-size: .82rem !important;
    text-align: left !important;
    line-height: 1.4 !important;
    padding: .6rem .9rem !important;
    transition: all .18s !important;
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: var(--amber) !important;
    color: var(--text) !important;
    background: var(--amber-glow) !important;
}

/* ── Chat input ────────────────────────────── */
[data-testid="stChatInput"] textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
    font-size: .9rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 2px var(--amber-glow) !important;
}

/* ── Expander ──────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stExpander"] summary { font-family: var(--font-head) !important; }

/* ── Slider ────────────────────────────────── */
[data-testid="stSlider"] > div > div > div { background: var(--amber) !important; }

/* ── Toggle ────────────────────────────────── */
[data-testid="stToggle"] > label { color: var(--text2) !important; font-family: var(--font-head) !important; }

/* ── Scrollbar ─────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--amber-dim); }

/* ── Hide Streamlit chrome ─────────────────── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Divider ───────────────────────────────── */
hr { border-color: var(--border) !important; margin: .8rem 0 !important; }

/* ── Info / warning banners ────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-mono) !important;
    font-size: .78rem !important;
}
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)

# JavaScript for copy-to-clipboard (injected once)
_COPY_JS = """
<script>
function auraCopy(id) {
    const el = document.getElementById(id);
    if (!el) return;
    navigator.clipboard.writeText(el.innerText).then(() => {
        const btn = document.querySelector('[onclick="auraCopy(\\''+id+'\\')"]');
        if (btn) { btn.innerText = '✓ Copied'; setTimeout(() => btn.innerText = 'Copy', 1500); }
    });
}
</script>
"""
st.markdown(_COPY_JS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# API CLIENT
# ─────────────────────────────────────────────

@st.cache_data(ttl=60)
def _fetch_providers() -> dict[str, list[str]]:
    """Fetch provider list from /providers; fallback to constants.py."""
    try:
        r = requests.get(f"{API_BASE}/providers", timeout=3)
        if r.ok:
            return r.json().get("providers", FALLBACK_PROVIDERS)
    except Exception:
        pass
    return FALLBACK_PROVIDERS


def _api_health() -> dict[str, Any] | None:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.ok else None
    except Exception:
        return None


def _api_ingest(files) -> dict[str, Any]:
    file_tuples = [(
        "files",
        (f.name, f.getvalue(), f.type or "application/octet-stream"),
    ) for f in files]
    r = requests.post(f"{API_BASE}/ingest", files=file_tuples, timeout=300)
    r.raise_for_status()
    return r.json()


def _build_query_payload(question: str, session_id: str) -> dict[str, Any]:
    """Assemble the common payload for both /query and /query/stream."""
    return {
        "question":   question,
        "top_k":      st.session_state.top_k,
        "provider":   st.session_state.provider,
        "model":      st.session_state.model,
        "api_key":    st.session_state.api_key,
        "session_id": session_id,
    }


def _api_query(question: str, session_id: str) -> dict[str, Any]:
    payload = _build_query_payload(question, session_id)
    r = requests.post(f"{API_BASE}/query", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def _api_stream(question: str, session_id: str) -> Generator[str, None, None]:
    """Real SSE streaming via /query/stream — yields one token at a time."""
    payload = _build_query_payload(question, session_id)
    with requests.post(
        f"{API_BASE}/query/stream",
        json=payload,
        stream=True,
        timeout=180,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                obj = json.loads(data_str)
                if "error" in obj:
                    yield f"\n\n❌ {obj['error']}"
                    break
                token = obj.get("token", "")
                if token:
                    yield token
            except json.JSONDecodeError:
                pass


def _api_clear_memory(session_id: str) -> None:
    try:
        requests.delete(f"{API_BASE}/memory/{session_id}", timeout=10)
    except Exception:
        pass


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

def _init_state() -> None:
    providers = _fetch_providers()
    defaults: dict[str, Any] = {
        "providers":   providers,
        "provider":    list(providers.keys())[0],
        "model":       list(providers.values())[0][0],
        "api_key":     "",
        "messages":    [],
        "key_valid":   None,
        "key_msg":     "",
        "streaming":   True,
        "top_k":       5,
        "session_id":  str(uuid.uuid4()),
        "ollama_model": "llama3",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─────────────────────────────────────────────
# LANGGRAPH PROGRESS COMPONENT
# ─────────────────────────────────────────────

_LG_STEPS = ["Rewrite", "Retrieve", "Grade", "Generate"]
_LG_ICONS = {"Rewrite": "✎", "Retrieve": "⌕", "Grade": "⊛", "Generate": "◈"}


def _render_pipeline_progress(active_step: str | None = None) -> None:
    """Render the 4-step LangGraph progress bar inline."""
    parts = []
    for i, step in enumerate(_LG_STEPS):
        if active_step is None:
            cls = ""
        elif step == active_step:
            cls = " active"
        elif _LG_STEPS.index(step) < _LG_STEPS.index(active_step):
            cls = " done"
        else:
            cls = ""
        icon = "✓" if " done" in cls else _LG_ICONS[step]
        parts.append(f'<span class="lg-step{cls}">{icon} {step}</span>')
        if i < len(_LG_STEPS) - 1:
            parts.append('<span class="lg-arrow">→</span>')

    st.markdown(
        f'<div class="lg-progress">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def _animate_pipeline_steps(placeholder: Any) -> None:
    """Block-level animation over the 4 LangGraph nodes while streaming."""
    for step in _LG_STEPS:
        with placeholder.container():
            _render_pipeline_progress(active_step=step)
        time.sleep(0.55)
    with placeholder.container():
        _render_pipeline_progress(active_step="Generate")  # keep on generate


# ─────────────────────────────────────────────
# CITATION RENDERER
# ─────────────────────────────────────────────

def _render_citations(sources: list[dict]) -> None:
    if not sources:
        return
    chips = '<div class="cite-row">'
    for src in sources:
        meta  = src.get("metadata", {})
        fname = meta.get("source", "Unknown").split("/")[-1].split("\\")[-1]
        page  = meta.get("page", "")
        label = f" · p.{page}" if page != "" else ""
        chips += f'<span class="cite-chip">📄 {fname}{label}</span>'
    chips += "</div>"
    st.markdown(chips, unsafe_allow_html=True)

    with st.expander(f"📚 {len(sources)} source(s)", expanded=False):
        for src in sources:
            meta    = src.get("metadata", {})
            fname   = meta.get("source", "Unknown").split("/")[-1]
            page    = meta.get("page", "—")
            snippet = src.get("content", "")[:320]
            st.markdown(
                f'<div class="src-card">'
                f'  <div class="src-meta"><span>📄 {fname}</span><span>Page {page}</span></div>'
                f'  <div class="src-snippet">{snippet}…</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        # ── Logo ──
        st.markdown(
            '<div class="sb-header">'
            '  <div class="sb-logo">◈ AuraRAG</div>'
            '  <div class="sb-tag">Advanced Unified Retrieval Architecture</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── System status ──
        with st.container():
            st.markdown('<div class="sb-label">System</div>', unsafe_allow_html=True)
            health = _api_health()
            if health:
                v  = health.get("version", "")
                vs = health.get("vector_store", "empty")
                vc = "pill-ok" if vs == "ready" else "pill-warn"
                st.markdown(
                    f'<span class="pill pill-ok">● API v{v}</span>'
                    f'<span class="pill {vc}">● DB {vs}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<span style="font-family:var(--font-mono);font-size:.65rem;color:var(--text3);">'
                    f'{health.get("docs_indexed","0")} chunks · '
                    f'{health.get("active_sessions","0")} session(s)</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<span class="pill pill-err">● API Offline</span>', unsafe_allow_html=True)

            kv    = st.session_state.key_valid
            k_cls = "pill-ok" if kv is True else ("pill-err" if kv is False else "pill-warn")
            k_lbl = "Key OK" if kv is True else ("Key Bad" if kv is False else "Key—")
            st.markdown(f'<span class="pill {k_cls}">● {k_lbl}</span>', unsafe_allow_html=True)
            st.markdown(
                f'<span style="font-family:var(--font-mono);font-size:.62rem;color:var(--text3);">'
                f'Session {st.session_state.session_id[:8]}…</span>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── AI Provider ──
        st.markdown('<div class="sb-label">AI Provider</div>', unsafe_allow_html=True)
        providers = st.session_state.providers

        provider = st.selectbox(
            "Provider",
            options=list(providers.keys()),
            index=list(providers.keys()).index(st.session_state.provider),
            label_visibility="collapsed",
        )
        if provider != st.session_state.provider:
            st.session_state.provider  = provider
            st.session_state.model     = providers[provider][0]
            st.session_state.key_valid = None
            st.session_state.key_msg   = ""

        # ── Model selector — special handling for Ollama ──
        if provider == "Local (Ollama)":
            ollama_input = st.text_input(
                "Ollama model name",
                value=st.session_state.ollama_model,
                placeholder="e.g. llama3, mistral, phi3…",
                label_visibility="collapsed",
                help="Enter any model pulled via `ollama pull <name>`.",
            )
            if ollama_input != st.session_state.ollama_model:
                st.session_state.ollama_model = ollama_input
            st.session_state.model     = ollama_input or "llama3"
            st.session_state.api_key   = ""
            st.session_state.key_valid = True
            st.session_state.key_msg   = ""
            st.markdown(
                '<span style="font-family:var(--font-mono);font-size:.7rem;color:#22c55e;">🦙 Local — no key needed</span>',
                unsafe_allow_html=True,
            )
        else:
            model_ids    = providers[provider]
            model_labels = [friendly_model_label(m) for m in model_ids]
            cur_idx      = model_ids.index(st.session_state.model) if st.session_state.model in model_ids else 0
            sel_label    = st.selectbox(
                "Model", options=model_labels, index=cur_idx, label_visibility="collapsed",
            )
            st.session_state.model = model_ids[model_labels.index(sel_label)]

            api_key = st.text_input(
                f"{provider} API Key",
                value=st.session_state.api_key,
                type="password",
                placeholder="Paste your API key…",
                label_visibility="collapsed",
                help="Stored in browser memory only. Never logged.",
            )
            if api_key != st.session_state.api_key:
                st.session_state.api_key = api_key
                valid, msg = validate_provider_config(provider, api_key)
                st.session_state.key_valid = valid
                st.session_state.key_msg   = msg

            if st.session_state.key_msg:
                col = "#22c55e" if st.session_state.key_valid else "#ef4444"
                st.markdown(
                    f'<span style="font-size:.68rem;font-family:var(--font-mono);color:{col};">'
                    f'{st.session_state.key_msg}</span>',
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Knowledge Base Upload ──
        st.markdown('<div class="sb-label">Knowledge Base</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "csv", "json", "xlsx", "parquet"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="Supported: PDF, TXT, CSV, JSON, XLSX, Parquet",
        )

        if uploaded:
            st.markdown(
                f'<span style="font-family:var(--font-mono);font-size:.68rem;color:var(--text3);">'
                f'{len(uploaded)} file(s) selected</span>',
                unsafe_allow_html=True,
            )

        if st.button("🚀  Ingest Documents", use_container_width=True):
            if not uploaded:
                st.warning("Select at least one file first.")
            else:
                prog = st.progress(0, text="Preparing…")
                try:
                    result   = _api_ingest(uploaded)
                    prog.progress(100, text="Done!")
                    dupes    = result.get("duplicates_skipped", 0)
                    dupe_note = f" ({dupes} duplicate(s) skipped)" if dupes else ""
                    st.success(
                        f"✅ {result['chunks_ingested']} chunks from "
                        f"{len(uploaded)} file(s){dupe_note}."
                    )
                except Exception as exc:
                    prog.empty()
                    st.error(f"Ingestion failed: {exc}")

        st.divider()

        # ── Advanced settings ──
        with st.expander("⚙️  Advanced", expanded=False):
            st.session_state.top_k = st.slider(
                "Top-K chunks",
                min_value=1, max_value=20,
                value=st.session_state.top_k,
                help="Number of document chunks to retrieve per query.",
            )
            st.session_state.streaming = st.toggle(
                "⚡ Streaming",
                value=st.session_state.streaming,
                help="Stream tokens as they arrive (recommended).",
            )

        st.divider()

        # ── Session controls ──
        if st.button("🆕  New Session", use_container_width=True):
            _api_clear_memory(st.session_state.session_id)
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages   = []
            st.rerun()

        if st.button("🗑️  Clear Chat", use_container_width=True):
            _api_clear_memory(st.session_state.session_id)
            st.session_state.messages = []
            st.rerun()

        if st.button("🔑  Clear API Key", use_container_width=True):
            st.session_state.api_key   = ""
            st.session_state.key_valid = None
            st.session_state.key_msg   = ""
            st.rerun()

        st.divider()
        st.markdown(
            '<div style="font-family:var(--font-mono);font-size:.62rem;color:var(--text3);text-align:center;">'
            'Built by <a href="https://github.com/thed700" style="color:var(--amber);text-decoration:none;">Akmal Raxmatov</a>'
            '</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# MAIN AREA HELPERS
# ─────────────────────────────────────────────

def _render_header() -> None:
    model_label = friendly_model_label(st.session_state.model)
    stream_html = (
        '<span class="stream-badge">⚡ streaming</span>'
        if st.session_state.streaming else ""
    )
    st.markdown(
        '<div class="chat-header">'
        '  <div class="chat-title">◈ AuraRAG</div>'
        f'  <div class="chat-subtitle">'
        f'    Hybrid Search · Re-ranking · Self-correction'
        f'    <span class="model-badge">◈ {st.session_state.provider} / {model_label}</span>'
        f'    {stream_html}'
        f'  </div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _guard_ready() -> bool:
    provider = st.session_state.provider
    if provider == "Local (Ollama)":
        return True
    if not st.session_state.api_key:
        st.info(f"👈 Paste your **{provider}** API key in the sidebar to start chatting.", icon="🔑")
        return False
    if st.session_state.key_valid is False:
        st.warning(f"The **{provider}** key looks invalid — check the sidebar.", icon="⚠️")
        return False
    return True


_STARTERS = [
    "What are the main topics covered in my documents?",
    "Summarise the key findings from the uploaded files.",
    "Are there any recommendations or conclusions I should know?",
    "What does the document say about [topic]?",
]


def _render_empty() -> None:
    st.markdown(
        '<div class="empty-state">'
        '  <div class="empty-icon">◈</div>'
        '  <div class="empty-title">Ask anything about your documents</div>'
        '  <div class="empty-sub">Upload files in the sidebar · Select your AI provider · Start chatting</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, prompt in enumerate(_STARTERS):
        with cols[i % 2]:
            if st.button(prompt, key=f"starter_{i}", use_container_width=True):
                _handle_input(prompt)
                st.rerun()


def _render_history() -> None:
    for msg in st.session_state.messages:
        avatar = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(avatar):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_citations(msg["sources"])


# ─────────────────────────────────────────────
# QUERY HANDLER
# ─────────────────────────────────────────────

def _handle_input(prompt: str) -> None:
    session_id = st.session_state.session_id

    # Add user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Show pipeline progress animation above the response
        progress_slot = st.empty()

        try:
            if st.session_state.streaming:
                # Animate steps while tokens stream in
                _animate_pipeline_steps(progress_slot)
                # Clear progress bar and stream response
                progress_slot.empty()
                full = st.write_stream(_api_stream(prompt, session_id))
                sources: list = []

            else:
                # Non-streaming: show animated steps then reveal answer
                _animate_pipeline_steps(progress_slot)
                result  = _api_query(prompt, session_id)
                full    = result.get("answer", "")
                sources = result.get("sources", [])
                progress_slot.empty()
                st.markdown(full)
                _render_citations(sources)

                # Optional observability trace
                trace = result.get("pipeline_trace", [])
                loops = result.get("reflect_loops", 0)
                if trace:
                    with st.expander("🔬 Pipeline trace", expanded=False):
                        st.markdown(
                            f'<span style="font-family:var(--font-mono);font-size:.72rem;color:var(--text3);">'
                            f'{"  →  ".join(trace)}'
                            f'{"  ·  " + str(loops) + " reflect loop(s)" if loops else ""}'
                            f'</span>',
                            unsafe_allow_html=True,
                        )

            st.session_state.messages.append(
                {"role": "assistant", "content": full, "sources": sources}
            )

        except requests.exceptions.ConnectionError:
            progress_slot.empty()
            err = "❌ Cannot reach the backend API. Is uvicorn running on port 8000?"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})

        except Exception as exc:
            progress_slot.empty()
            err = f"❌ {exc}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main() -> None:
    _render_sidebar()
    _render_header()

    if not st.session_state.messages:
        _render_empty()
    else:
        _render_history()

    if _guard_ready():
        if prompt := st.chat_input("Ask a question about your documents…"):
            _handle_input(prompt)


if __name__ == "__main__":
    main()
