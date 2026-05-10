"""
ui.py - Streamlit Chat Interface
Author: Akmal Raxmatov (github: thed700)

FIXES applied (v3.0.0):
  BUG-10: Replaced fake _stream_text() word-replay with direct answer display.
          True SSE streaming from the backend is tracked as a v3.1 milestone.
  Rebranded all visible text from NeuralDocs -> AuraRAG.
  CORS: UI now runs on http://localhost:8501 which matches the explicit
        ALLOWED_ORIGINS list in main.py (no more wildcard needed).
"""

import os
import requests
import streamlit as st
from typing import Any

from app.engine import PROVIDER_MODELS, validate_provider_config

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AuraRAG — Enterprise Knowledge Base",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

# ─────────────────────────────────────────────
# VISUAL DESIGN SYSTEM
# ─────────────────────────────────────────────

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:          #0f0f11;
    --surface:     #17171a;
    --surface2:    #1e1e23;
    --border:      #2a2a32;
    --amber:       #f5a623;
    --amber-dim:   #c47d0e;
    --amber-glow:  rgba(245,166,35,0.12);
    --text:        #e8e8ec;
    --text-muted:  #7a7a8a;
    --green:       #22c55e;
    --red:         #ef4444;
    --blue:        #60a5fa;
    --radius:      10px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

.stApp { background: var(--bg); }

[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text); }

[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text-muted);
    font-size: 0.82rem;
    transition: all 0.18s ease;
    padding: 0.45rem 1rem;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--amber);
    color: var(--amber);
    background: var(--amber-glow);
}

.ar-header { display: flex; align-items: center; gap: 12px; padding: 1.2rem 0 0.4rem 0; }
.ar-logo { font-family: 'DM Serif Display', serif; font-size: 2rem; color: var(--amber); letter-spacing: -0.02em; line-height: 1; }
.ar-tagline { font-size: 0.75rem; color: var(--text-muted); font-family: 'DM Mono', monospace; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 2px; }

.status-pill { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 20px; font-size: 0.72rem; font-family: 'DM Mono', monospace; font-weight: 500; margin: 2px 0; }
.status-ok   { background: rgba(34,197,94,0.12);  color: #22c55e; border: 1px solid rgba(34,197,94,0.25); }
.status-err  { background: rgba(239,68,68,0.12);   color: #ef4444; border: 1px solid rgba(239,68,68,0.25); }
.status-warn { background: rgba(245,166,35,0.12);  color: #f5a623; border: 1px solid rgba(245,166,35,0.25); }

[data-testid="stChatMessage"] { border-radius: var(--radius) !important; margin-bottom: 0.6rem !important; border: 1px solid var(--border) !important; }

.citation-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.citation-chip { display: inline-flex; align-items: center; gap: 5px; padding: 3px 9px; background: var(--surface2); border: 1px solid var(--border); border-radius: 20px; font-family: 'DM Mono', monospace; font-size: 0.70rem; color: var(--text-muted); cursor: pointer; transition: border-color 0.15s ease, color 0.15s ease; }
.citation-chip:hover { border-color: var(--amber); color: var(--amber); }

.source-card { background: var(--surface2); border-left: 3px solid var(--amber); border-radius: 0 var(--radius) var(--radius) 0; padding: 10px 14px; margin-bottom: 8px; font-size: 0.80rem; line-height: 1.55; }
.source-meta { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: var(--amber); margin-bottom: 5px; display: flex; gap: 12px; }
.source-snippet { color: var(--text-muted); }

.prompt-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 1.5rem; max-width: 600px; margin-left: auto; margin-right: auto; }
.prompt-card { background: var(--surface2); border: 1px solid var(--border); border-radius: var(--radius); padding: 12px 14px; font-size: 0.80rem; color: var(--text-muted); cursor: pointer; transition: border-color 0.15s ease, color 0.15s ease; line-height: 1.4; }
.prompt-card:hover { border-color: var(--amber); color: var(--text); }

[data-testid="stChatInput"] textarea { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; color: var(--text) !important; font-family: 'DM Sans', sans-serif !important; }
[data-testid="stChatInput"] textarea:focus { border-color: var(--amber) !important; box-shadow: 0 0 0 2px var(--amber-glow) !important; }

hr { border-color: var(--border) !important; }
[data-testid="stExpander"] { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }
[data-testid="stSpinner"] { color: var(--amber) !important; }

.provider-badge { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: var(--amber); background: var(--amber-glow); border: 1px solid var(--amber-dim); border-radius: 4px; padding: 1px 6px; }

::-webkit-scrollbar       { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--amber-dim); }

#MainMenu, footer, header { visibility: hidden; }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

def _init_session_state() -> None:
    defaults = {
        "provider":  "OpenAI",
        "model":     PROVIDER_MODELS["OpenAI"][0],
        "api_key":   "",
        "messages":  [],
        "key_valid": None,
        "key_msg":   "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()


# ─────────────────────────────────────────────
# API CLIENT
# ─────────────────────────────────────────────

def _api_health() -> dict[str, Any] | None:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.ok else None
    except Exception:
        return None


def _api_ingest(files) -> dict[str, Any]:
    file_tuples = [("files", (f.name, f.getvalue(), f.type)) for f in files]
    r = requests.post(f"{API_BASE}/ingest", files=file_tuples, timeout=120)
    r.raise_for_status()
    return r.json()


def _api_query(question: str) -> dict[str, Any]:
    payload = {
        "question": question,
        "top_k": 5,
        "provider": st.session_state.provider,
        "model":    st.session_state.model,
        "api_key":  st.session_state.api_key,
    }
    r = requests.post(f"{API_BASE}/query", json=payload, timeout=90)
    r.raise_for_status()
    return r.json()


def _api_clear_memory() -> None:
    requests.delete(f"{API_BASE}/memory", timeout=10)


# ─────────────────────────────────────────────
# CITATION RENDERER
# ─────────────────────────────────────────────

def _render_citations(sources: list[dict]) -> None:
    if not sources:
        return
    chips_html = '<div class="citation-row">'
    for src in sources:
        meta = src.get("metadata", {})
        fname = meta.get("source", "Unknown").split("/")[-1].split("\\")[-1]
        page = meta.get("page", "")
        page_label = f" · p.{page}" if page != "" else ""
        chips_html += f'<span class="citation-chip">📄 {fname}{page_label}</span>'
    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)

    with st.expander(f"📚 View {len(sources)} source snippet(s)", expanded=False):
        for src in sources:
            meta    = src.get("metadata", {})
            fname   = meta.get("source", "Unknown").split("/")[-1]
            page    = meta.get("page", "—")
            snippet = src.get("content", "")[:320]
            st.markdown(
                f'<div class="source-card">'
                f'  <div class="source-meta"><span>📄 {fname}</span><span>Page {page}</span></div>'
                f'  <div class="source-snippet">{snippet}…</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _friendly_model_label(model_id: str) -> str:
    label_map = {
        "claude-haiku-4-5-20251001":  "claude-haiku-4-5",
        "claude-3-5-sonnet-20241022": "claude-3.5-sonnet",
        "claude-3-opus-20240229":     "claude-3-opus",
    }
    return label_map.get(model_id, model_id)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div style="padding:0.6rem 0 1rem 0;">'
            '  <span style="font-family:\'DM Serif Display\',serif;font-size:1.4rem;'
            'color:#f5a623;letter-spacing:-0.02em;">◈ AuraRAG</span>'
            '  <div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
            'color:#7a7a8a;text-transform:uppercase;letter-spacing:0.1em;margin-top:2px;">'
            'Advanced Unified Retrieval Architecture</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**System Status**")
        health = _api_health()
        if health:
            version = health.get("version", "")
            version_label = f" v{version}" if version else ""
            st.markdown(
                f'<span class="status-pill status-ok">● API Online{version_label}</span>',
                unsafe_allow_html=True,
            )
            vs_status = health.get("vector_store", "empty")
            vs_class  = "status-ok" if vs_status == "ready" else "status-warn"
            st.markdown(f'<span class="status-pill {vs_class}">● ChromaDB {vs_status}</span>', unsafe_allow_html=True)
            st.markdown(
                f'<span style="font-family:DM Mono,monospace;font-size:0.68rem;color:#7a7a8a;">'
                f'{health.get("docs_indexed","0")} chunks indexed</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<span class="status-pill status-err">● API Offline</span>', unsafe_allow_html=True)

        if st.session_state.key_valid is True:
            st.markdown('<span class="status-pill status-ok">● LLM Key Valid</span>', unsafe_allow_html=True)
        elif st.session_state.key_valid is False:
            st.markdown('<span class="status-pill status-err">● LLM Key Invalid</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill status-warn">● LLM Key Not Set</span>', unsafe_allow_html=True)

        st.divider()

        st.markdown("**🧠 AI Provider**")
        provider = st.selectbox(
            "Provider",
            options=list(PROVIDER_MODELS.keys()),
            index=list(PROVIDER_MODELS.keys()).index(st.session_state.provider),
            label_visibility="collapsed",
        )

        if provider != st.session_state.provider:
            st.session_state.provider  = provider
            st.session_state.model     = PROVIDER_MODELS[provider][0]
            st.session_state.key_valid = None
            st.session_state.key_msg   = ""

        model_ids    = PROVIDER_MODELS[provider]
        model_labels = [_friendly_model_label(m) for m in model_ids]
        current_idx  = (
            model_ids.index(st.session_state.model)
            if st.session_state.model in model_ids
            else 0
        )
        selected_label = st.selectbox(
            "Model",
            options=model_labels,
            index=current_idx,
            label_visibility="collapsed",
        )
        st.session_state.model = model_ids[model_labels.index(selected_label)]

        if provider == "Local (Ollama)":
            st.info("🦙 Ollama runs locally — no API key needed.")
            st.session_state.api_key   = ""
            st.session_state.key_valid = True
            st.session_state.key_msg   = "Local model — no key required."
        else:
            api_key = st.text_input(
                f"{provider} API Key",
                value=st.session_state.api_key,
                type="password",
                placeholder="Paste your API key…",
                help=(
                    "Your key is stored only in browser session memory. "
                    "It is never saved to disk, logged, or transmitted "
                    "beyond the local FastAPI backend."
                ),
            )
            if api_key != st.session_state.api_key:
                st.session_state.api_key   = api_key
                valid, msg = validate_provider_config(provider, api_key)
                st.session_state.key_valid = valid
                st.session_state.key_msg   = msg

            if st.session_state.key_msg:
                colour = "#22c55e" if st.session_state.key_valid else "#ef4444"
                st.markdown(
                    f'<span style="font-size:0.70rem;font-family:DM Mono,monospace;'
                    f'color:{colour};">{st.session_state.key_msg}</span>',
                    unsafe_allow_html=True,
                )

        st.divider()

        st.markdown("**📄 Knowledge Base**")
        uploaded = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if st.button("🚀 Ingest Documents", use_container_width=True):
            if not uploaded:
                st.warning("Select at least one file first.")
            else:
                with st.spinner("Chunking & indexing…"):
                    try:
                        result = _api_ingest(uploaded)
                        st.success(
                            f"✅ {result['chunks_ingested']} chunks from {len(uploaded)} file(s)."
                        )
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")

        st.divider()

        if st.button("🗑️ Clear Conversation", use_container_width=True):
            _api_clear_memory()
            st.session_state.messages = []
            st.rerun()

        if st.button("🔑 Clear API Key", use_container_width=True):
            st.session_state.api_key   = ""
            st.session_state.key_valid = None
            st.session_state.key_msg   = ""
            st.rerun()

        st.divider()
        st.caption("Built by **Akmal Raxmatov** · [GitHub](https://github.com/thed700)")


# ─────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────

def _render_header() -> None:
    provider_html = (
        f'<span class="provider-badge">'
        f'{st.session_state.provider} / {_friendly_model_label(st.session_state.model)}'
        f'</span>'
    )
    st.markdown(
        f'<div class="ar-header">'
        f'  <div>'
        f'    <div class="ar-logo">◈ AuraRAG</div>'
        f'    <div class="ar-tagline">Hybrid Search · Re-ranking · {provider_html}</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_message_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_citations(msg["sources"])


def _guard_llm_ready() -> bool:
    provider = st.session_state.provider
    if provider == "Local (Ollama)":
        return True
    if not st.session_state.api_key:
        st.info(f"👈 Paste your **{provider}** API key in the sidebar to start chatting.", icon="🔑")
        return False
    if st.session_state.key_valid is False:
        st.warning(f"The API key for **{provider}** doesn't look right. Check the sidebar.", icon="⚠️")
        return False
    return True


STARTER_PROMPTS = [
    "What are the main topics covered in my documents?",
    "Summarise the key findings from the uploaded files.",
    "What does the document say about [topic]?",
    "List any recommendations or conclusions mentioned.",
]


def _render_empty_state() -> None:
    st.markdown(
        '<div style="text-align:center;padding:2.5rem 0 1rem 0;color:#7a7a8a;">'
        '  <div style="font-size:2.5rem;margin-bottom:0.5rem;">◈</div>'
        '  <div style="font-family:DM Serif Display,serif;font-size:1.2rem;'
        'color:#e8e8ec;margin-bottom:0.4rem;">Ask anything about your documents</div>'
        '  <div style="font-size:0.82rem;">'
        'Upload files via the sidebar · Select your AI provider · Start chatting'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, prompt in enumerate(STARTER_PROMPTS):
        with cols[i % 2]:
            if st.button(prompt, key=f"starter_{i}", use_container_width=True):
                _handle_user_input(prompt)
                st.rerun()


def _handle_user_input(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            result  = _api_query(prompt)
            answer  = result.get("answer", "")
            sources = result.get("sources", [])
            # FIX BUG-10: display answer directly — no fake word-by-word replay.
            # True SSE streaming is planned for v3.1.
            st.markdown(answer)
            _render_citations(sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except requests.exceptions.ConnectionError:
            err = (
                "❌ Cannot reach the backend API. "
                "Make sure `uvicorn app.main:app` is running on port 8000."
            )
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
        except Exception as e:
            err = f"❌ Error: {e}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main() -> None:
    _render_sidebar()
    _render_header()
    st.divider()

    if not st.session_state.messages:
        _render_empty_state()
    else:
        _render_message_history()

    if _guard_llm_ready():
        if prompt := st.chat_input(
            "Ask a question about your documents… (Enter to send)",
            key="chat_input",
        ):
            _handle_user_input(prompt)


if __name__ == "__main__":
    main()
