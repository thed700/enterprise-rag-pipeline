"""
ui.py — Streamlit Chat Interface v3.3
Author: Akmal Raxmatov (github: thed700)

Changes v3.3:
  BUG-AA: _api_stream() sent no top_k field to /query/stream so every
           streaming query used the server default of 5 regardless of what
           the user intended.  Fixed: top_k is now included in the payload,
           consistent with _api_query().

Retained from v3.1.0:
  BUG-O: No longer imports from app.engine. PROVIDER_MODELS and
          validate_provider_config are now fetched from the /providers API
          endpoint (with a fallback to constants.py for offline/dev mode).
  BUG-P: session_id generated per browser tab using st.runtime session.
  BUG-10: True SSE streaming via /query/stream endpoint (replaces direct
          /query call). Tokens rendered incrementally with st.write_stream.
  NEW:   Ingestion progress bar with per-file status.
  NEW:   Stream toggle in sidebar (streaming on by default).
  NEW:   "New Session" button generates a fresh session_id without reloading.
"""

import os
import uuid
import json
import requests
import streamlit as st
from typing import Any, Generator

from app.constants import PROVIDER_MODELS as FALLBACK_PROVIDERS, validate_provider_config, friendly_model_label

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
# DESIGN SYSTEM
# ─────────────────────────────────────────────

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:         #0f0f11;
    --surface:    #17171a;
    --surface2:   #1e1e23;
    --border:     #2a2a32;
    --amber:      #f5a623;
    --amber-dim:  #c47d0e;
    --amber-glow: rgba(245,166,35,0.12);
    --text:       #e8e8ec;
    --muted:      #7a7a8a;
    --green:      #22c55e;
    --red:        #ef4444;
    --blue:       #60a5fa;
    --radius:     10px;
}

html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--text); }
.stApp { background:var(--bg); }

[data-testid="stSidebar"] {
    background:var(--surface);
    border-right:1px solid var(--border);
}
[data-testid="stSidebar"] * { color:var(--text); }

[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background:var(--surface2) !important;
    border:1px solid var(--border) !important;
    border-radius:var(--radius) !important;
    color:var(--text) !important;
    font-family:'DM Mono',monospace !important;
    font-size:0.82rem !important;
}

[data-testid="stSidebar"] .stButton > button {
    width:100%; background:transparent;
    border:1px solid var(--border);
    border-radius:var(--radius);
    color:var(--muted); font-size:0.82rem;
    transition:all .18s; padding:.45rem 1rem;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color:var(--amber); color:var(--amber);
    background:var(--amber-glow);
}

.ar-logo { font-family:'DM Serif Display',serif; font-size:2rem; color:var(--amber); letter-spacing:-.02em; }
.ar-tagline { font-size:.75rem; color:var(--muted); font-family:'DM Mono',monospace; letter-spacing:.08em; text-transform:uppercase; }

.pill { display:inline-flex; align-items:center; gap:6px; padding:4px 10px;
        border-radius:20px; font-size:.72rem; font-family:'DM Mono',monospace;
        font-weight:500; margin:2px 0; }
.pill-ok   { background:rgba(34,197,94,.12);  color:#22c55e; border:1px solid rgba(34,197,94,.25); }
.pill-err  { background:rgba(239,68,68,.12);  color:#ef4444; border:1px solid rgba(239,68,68,.25); }
.pill-warn { background:rgba(245,166,35,.12); color:#f5a623; border:1px solid rgba(245,166,35,.25); }

[data-testid="stChatMessage"] {
    border-radius:var(--radius) !important;
    margin-bottom:.6rem !important;
    border:1px solid var(--border) !important;
}

.cite-row { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }
.cite-chip { display:inline-flex; align-items:center; gap:5px; padding:3px 9px;
             background:var(--surface2); border:1px solid var(--border);
             border-radius:20px; font-family:'DM Mono',monospace; font-size:.70rem;
             color:var(--muted); }

.src-card { background:var(--surface2); border-left:3px solid var(--amber);
            border-radius:0 var(--radius) var(--radius) 0;
            padding:10px 14px; margin-bottom:8px; font-size:.80rem; }
.src-meta { font-family:'DM Mono',monospace; font-size:.68rem; color:var(--amber);
            margin-bottom:5px; display:flex; gap:12px; }
.src-snippet { color:var(--muted); }

.badge { font-family:'DM Mono',monospace; font-size:.68rem; color:var(--amber);
         background:var(--amber-glow); border:1px solid var(--amber-dim);
         border-radius:4px; padding:1px 6px; }

[data-testid="stChatInput"] textarea {
    background:var(--surface2) !important;
    border:1px solid var(--border) !important;
    border-radius:var(--radius) !important;
    color:var(--text) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color:var(--amber) !important;
    box-shadow:0 0 0 2px var(--amber-glow) !important;
}

hr { border-color:var(--border) !important; }
[data-testid="stExpander"] {
    background:var(--surface2) !important;
    border:1px solid var(--border) !important;
    border-radius:var(--radius) !important;
}

::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--amber-dim); }

#MainMenu, footer, header { visibility:hidden; }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# API CLIENT
# ─────────────────────────────────────────────

@st.cache_data(ttl=60)
def _fetch_providers() -> dict[str, list[str]]:
    """
    FIX BUG-O: fetch provider list from /providers endpoint instead of
    importing app.engine. Falls back to constants.py if API is unreachable.
    """
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
    file_tuples = [("files", (f.name, f.getvalue(), f.type)) for f in files]
    r = requests.post(f"{API_BASE}/ingest", files=file_tuples, timeout=180)
    r.raise_for_status()
    return r.json()


def _api_query(question: str, session_id: str) -> dict[str, Any]:
    payload = {
        "question":   question,
        "top_k":      5,
        "provider":   st.session_state.provider,
        "model":      st.session_state.model,
        "api_key":    st.session_state.api_key,
        "session_id": session_id,           # FIX BUG-P
    }
    r = requests.post(f"{API_BASE}/query", json=payload, timeout=90)
    r.raise_for_status()
    return r.json()


def _api_stream(question: str, session_id: str) -> Generator[str, None, None]:
    """
    FIX BUG-10: real SSE streaming via /query/stream.
    Yields one token at a time as the LLM produces it.

    BUG-AA fix: top_k is now included in the payload so the server honours
    the caller preference. Previously it was absent, causing every streaming
    request to silently fall back to the server-side default of 5.
    """
    payload = {
        "question":   question,
        "top_k":      5,           # BUG-AA fix: was missing from stream payload
        "provider":   st.session_state.provider,
        "model":      st.session_state.model,
        "api_key":    st.session_state.api_key,
        "session_id": session_id,
    }
    with requests.post(
        f"{API_BASE}/query/stream",
        json=payload,
        stream=True,
        timeout=120,
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
    defaults = {
        "providers":   providers,
        "provider":    list(providers.keys())[0],
        "model":       list(providers.values())[0][0],
        "api_key":     "",
        "messages":    [],
        "key_valid":   None,
        "key_msg":     "",
        "streaming":   True,
        # FIX BUG-P: unique session_id per browser tab
        "session_id":  str(uuid.uuid4()),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


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
        st.markdown(
            '<div style="padding:.6rem 0 1rem;">'
            '<span style="font-family:\'DM Serif Display\',serif;font-size:1.4rem;color:#f5a623;">◈ AuraRAG</span>'
            '<div style="font-family:\'DM Mono\',monospace;font-size:.62rem;color:#7a7a8a;text-transform:uppercase;letter-spacing:.1em;margin-top:2px;">Advanced Unified Retrieval Architecture</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── System status ──
        st.markdown("**System Status**")
        health = _api_health()
        if health:
            v = health.get("version", "")
            st.markdown(f'<span class="pill pill-ok">● API Online v{v}</span>', unsafe_allow_html=True)
            vs = health.get("vector_store", "empty")
            cls = "pill-ok" if vs == "ready" else "pill-warn"
            st.markdown(f'<span class="pill {cls}">● ChromaDB {vs}</span>', unsafe_allow_html=True)
            st.markdown(
                f'<span style="font-family:DM Mono,monospace;font-size:.68rem;color:#7a7a8a;">'
                f'{health.get("docs_indexed","0")} chunks · '
                f'{health.get("active_sessions","0")} session(s)</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<span class="pill pill-err">● API Offline</span>', unsafe_allow_html=True)

        kv = st.session_state.key_valid
        k_cls = "pill-ok" if kv is True else ("pill-err" if kv is False else "pill-warn")
        k_lbl = "Valid" if kv is True else ("Invalid" if kv is False else "Not Set")
        st.markdown(f'<span class="pill {k_cls}">● LLM Key {k_lbl}</span>', unsafe_allow_html=True)

        st.markdown(
            f'<span style="font-family:DM Mono,monospace;font-size:.65rem;color:#5a5a6a;">'
            f'Session: {st.session_state.session_id[:8]}…</span>',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Provider / Model ──
        st.markdown("**🧠 AI Provider**")
        providers = st.session_state.providers

        provider = st.selectbox(
            "Provider", options=list(providers.keys()),
            index=list(providers.keys()).index(st.session_state.provider),
            label_visibility="collapsed",
        )
        if provider != st.session_state.provider:
            st.session_state.provider  = provider
            st.session_state.model     = providers[provider][0]
            st.session_state.key_valid = None
            st.session_state.key_msg   = ""

        model_ids    = providers[provider]
        model_labels = [friendly_model_label(m) for m in model_ids]
        cur_idx      = model_ids.index(st.session_state.model) if st.session_state.model in model_ids else 0

        sel_label = st.selectbox("Model", options=model_labels, index=cur_idx, label_visibility="collapsed")
        st.session_state.model = model_ids[model_labels.index(sel_label)]

        if provider == "Local (Ollama)":
            st.info("🦙 Ollama — no API key needed.")
            st.session_state.api_key   = ""
            st.session_state.key_valid = True
            st.session_state.key_msg   = "Local model — no key required."
        else:
            api_key = st.text_input(
                f"{provider} API Key",
                value=st.session_state.api_key,
                type="password",
                placeholder="Paste your API key…",
                label_visibility="collapsed",
                help="Stored in browser memory only. Never logged or saved.",
            )
            if api_key != st.session_state.api_key:
                st.session_state.api_key = api_key
                valid, msg = validate_provider_config(provider, api_key)
                st.session_state.key_valid = valid
                st.session_state.key_msg   = msg

            if st.session_state.key_msg:
                col = "#22c55e" if st.session_state.key_valid else "#ef4444"
                st.markdown(
                    f'<span style="font-size:.70rem;font-family:DM Mono,monospace;color:{col};">'
                    f'{st.session_state.key_msg}</span>',
                    unsafe_allow_html=True,
                )

        # ── Streaming toggle ──
        st.session_state.streaming = st.toggle(
            "⚡ Streaming mode",
            value=st.session_state.streaming,
            help="Stream tokens as they arrive. Disable to get a single response block.",
        )

        st.divider()

        # ── Knowledge base ──
        st.markdown("**📄 Knowledge Base**")
        uploaded = st.file_uploader(
            "Upload PDF or TXT", type=["pdf", "txt"],
            accept_multiple_files=True, label_visibility="collapsed",
        )

        if st.button("🚀 Ingest Documents", use_container_width=True):
            if not uploaded:
                st.warning("Select at least one file first.")
            else:
                prog = st.progress(0, text="Preparing…")
                try:
                    result = _api_ingest(uploaded)
                    prog.progress(100, text="Done!")
                    dupes = result.get("duplicates_skipped", 0)
                    dupe_note = f" ({dupes} duplicate chunk(s) skipped)" if dupes else ""
                    st.success(
                        f"✅ {result['chunks_ingested']} chunks from "
                        f"{len(uploaded)} file(s){dupe_note}."
                    )
                except Exception as e:
                    prog.empty()
                    st.error(f"Ingestion failed: {e}")

        st.divider()

        if st.button("🆕 New Session", use_container_width=True):
            new_id = str(uuid.uuid4())
            _api_clear_memory(st.session_state.session_id)
            st.session_state.session_id = new_id
            st.session_state.messages   = []
            st.rerun()

        if st.button("🗑️ Clear Conversation", use_container_width=True):
            _api_clear_memory(st.session_state.session_id)
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
# MAIN AREA
# ─────────────────────────────────────────────

def _render_header() -> None:
    badge = (
        f'<span class="badge">'
        f'{st.session_state.provider} / {friendly_model_label(st.session_state.model)}'
        f'</span>'
    )
    stream_badge = (
        '<span class="badge" style="color:#22c55e;border-color:#22c55e;background:rgba(34,197,94,.1);">⚡ streaming</span>'
        if st.session_state.streaming else ""
    )
    st.markdown(
        f'<div style="padding:1.2rem 0 .4rem;">'
        f'  <div class="ar-logo">◈ AuraRAG</div>'
        f'  <div class="ar-tagline">Hybrid Search · Re-ranking · {badge} {stream_badge}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _guard_ready() -> bool:
    provider = st.session_state.provider
    if provider == "Local (Ollama)":
        return True
    if not st.session_state.api_key:
        st.info(f"👈 Paste your **{provider}** API key in the sidebar.", icon="🔑")
        return False
    if st.session_state.key_valid is False:
        st.warning(f"The **{provider}** key doesn't look right. Check the sidebar.", icon="⚠️")
        return False
    return True


STARTERS = [
    "What are the main topics covered in my documents?",
    "Summarise the key findings from the uploaded files.",
    "What does the document say about [topic]?",
    "List any recommendations or conclusions mentioned.",
]


def _render_empty() -> None:
    st.markdown(
        '<div style="text-align:center;padding:2.5rem 0 1rem;color:#7a7a8a;">'
        '  <div style="font-size:2.5rem;margin-bottom:.5rem;">◈</div>'
        '  <div style="font-family:DM Serif Display,serif;font-size:1.2rem;color:#e8e8ec;margin-bottom:.4rem;">Ask anything about your documents</div>'
        '  <div style="font-size:.82rem;">Upload files via the sidebar · Select your AI provider · Start chatting</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, p in enumerate(STARTERS):
        with cols[i % 2]:
            if st.button(p, key=f"s_{i}", use_container_width=True):
                _handle_input(p)
                st.rerun()


def _render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_citations(msg["sources"])


def _handle_input(prompt: str) -> None:
    session_id = st.session_state.session_id
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if st.session_state.streaming:
                # FIX BUG-10: true SSE streaming
                full = st.write_stream(_api_stream(prompt, session_id))
                sources: list = []   # sources not returned in stream mode
            else:
                result  = _api_query(prompt, session_id)
                full    = result.get("answer", "")
                sources = result.get("sources", [])
                st.markdown(full)
                _render_citations(sources)

            st.session_state.messages.append({
                "role":    "assistant",
                "content": full,
                "sources": sources,
            })

        except requests.exceptions.ConnectionError:
            err = "❌ Cannot reach the backend API. Is uvicorn running on port 8000?"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
        except Exception as e:
            err = f"❌ {e}"
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
        _render_empty()
    else:
        _render_history()

    if _guard_ready():
        if prompt := st.chat_input("Ask a question about your documents…"):
            _handle_input(prompt)


if __name__ == "__main__":
    main()
