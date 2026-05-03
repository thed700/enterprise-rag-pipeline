"""
ui.py - Streamlit Chat Interface (SaaS Edition)
Author: Akmal Raxmatov (github: thed700)

Run: streamlit run app/ui.py

Architecture:
  - All LLM credentials live ONLY in st.session_state — never on disk.
  - Streaming responses via LangChain StreamingStdOutCallbackHandler +
    Streamlit's st.write_stream pattern.
  - Hybrid Search + Cross-Encoder re-ranking preserved from engine.py.
  - Source citations rendered as collapsible chips below each reply.
"""

import time
import requests
import streamlit as st
from typing import Any, Generator

from app.engine import PROVIDER_MODELS, validate_provider_config

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be the very first st call)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="NeuralDocs — Enterprise RAG",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# ─────────────────────────────────────────────
# VISUAL DESIGN SYSTEM
# Dark-mode, editorial aesthetic with amber accent
# ─────────────────────────────────────────────

STYLES = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
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
    --user-bubble: #1c2333;
    --ai-bubble:   #17171a;
    --radius:      10px;
}

/* ── Global overrides ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

.stApp {
    background: var(--bg);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

/* ── Sidebar select / input ── */
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Sidebar buttons ── */
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

/* ── Primary CTA button (Ingest) ── */
[data-testid="stSidebar"] .stButton:first-of-type > button {
    background: var(--amber);
    border-color: var(--amber);
    color: #0f0f11;
    font-weight: 600;
}

[data-testid="stSidebar"] .stButton:first-of-type > button:hover {
    background: var(--amber-dim);
    border-color: var(--amber-dim);
    color: #0f0f11;
}

/* ── Main header ── */
.nd-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 1.2rem 0 0.4rem 0;
}

.nd-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--amber);
    letter-spacing: -0.02em;
    line-height: 1;
}

.nd-tagline {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    margin: 2px 0;
}

.status-ok   { background: rgba(34,197,94,0.12);  color: #22c55e; border: 1px solid rgba(34,197,94,0.25); }
.status-err  { background: rgba(239,68,68,0.12);   color: #ef4444; border: 1px solid rgba(239,68,68,0.25); }
.status-warn { background: rgba(245,166,35,0.12);  color: #f5a623; border: 1px solid rgba(245,166,35,0.25); }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: var(--radius) !important;
    margin-bottom: 0.6rem !important;
    border: 1px solid var(--border) !important;
}

[data-testid="stChatMessage"][data-testid*="user"] {
    background: var(--user-bubble) !important;
}

[data-testid="stChatMessage"][data-testid*="assistant"] {
    background: var(--ai-bubble) !important;
}

/* ── Source citations chip row ── */
.citation-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
}

.citation-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 9px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.70rem;
    color: var(--text-muted);
    cursor: pointer;
    transition: border-color 0.15s ease, color 0.15s ease;
}

.citation-chip:hover {
    border-color: var(--amber);
    color: var(--amber);
}

/* ── Source expander ── */
.source-card {
    background: var(--surface2);
    border-left: 3px solid var(--amber);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.80rem;
    line-height: 1.55;
}

.source-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--amber);
    margin-bottom: 5px;
    display: flex;
    gap: 12px;
}

.source-snippet {
    color: var(--text-muted);
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 2px var(--amber-glow) !important;
}

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--amber) !important; }

/* ── Provider badge ── */
.provider-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--amber);
    background: var(--amber-glow);
    border: 1px solid var(--amber-dim);
    border-radius: 4px;
    padding: 1px 6px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar       { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--amber-dim); }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE BOOTSTRAP
# Keys are NEVER written to disk or .env.
# They live exclusively in st.session_state,
# which is per-tab and garbage-collected on
# browser close / tab refresh.
# ─────────────────────────────────────────────

def _init_session_state() -> None:
    """Initialise all session-state keys with safe defaults."""
    defaults = {
        # LLM config — ephemeral, in-memory only
        "provider":     "OpenAI",
        "model":        PROVIDER_MODELS["OpenAI"][0],
        "api_key":      "",          # ← the only place the key ever lives
        # Chat history for rendering
        "messages":     [],
        # Key validation result cache
        "key_valid":    None,        # None | True | False
        "key_msg":      "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()


# ─────────────────────────────────────────────
# API CLIENT HELPERS
# ─────────────────────────────────────────────

def _api_health() -> dict[str, Any] | None:
    """Ping the FastAPI backend health endpoint."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.ok else None
    except Exception:
        return None


def _api_ingest(files) -> dict[str, Any]:
    """POST files to the /ingest endpoint."""
    file_tuples = [("files", (f.name, f.getvalue(), f.type)) for f in files]
    r = requests.post(f"{API_BASE}/ingest", files=file_tuples, timeout=120)
    r.raise_for_status()
    return r.json()


def _api_query(question: str) -> dict[str, Any]:
    """POST a question with provider config to /query."""
    payload = {
        "question": question,
        "top_k": 5,
        # LLM config forwarded from session_state — not from disk
        "provider": st.session_state.provider,
        "model":    st.session_state.model,
        "api_key":  st.session_state.api_key,
    }
    r = requests.post(f"{API_BASE}/query", json=payload, timeout=90)
    r.raise_for_status()
    return r.json()


def _api_clear_memory() -> None:
    """DELETE /memory to wipe server-side conversation buffer."""
    requests.delete(f"{API_BASE}/memory", timeout=10)


# ─────────────────────────────────────────────
# STREAMING SIMULATION
# When the backend returns a full string (non-streaming
# FastAPI mode), we simulate word-by-word output in the
# UI for the ChatGPT feel. Swap for a true SSE/websocket
# stream when the backend supports it.
# ─────────────────────────────────────────────

def _stream_text(text: str, delay: float = 0.018) -> Generator[str, None, None]:
    """Yield words with a small delay to simulate streaming."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


# ─────────────────────────────────────────────
# SOURCE CITATION RENDERER
# ─────────────────────────────────────────────

def _render_citations(sources: list[dict]) -> None:
    """
    Render source citations as compact chips + an expander with snippets.
    Each chip shows filename and page (when available).
    """
    if not sources:
        return

    # Build chip labels
    chips_html = '<div class="citation-row">'
    for i, src in enumerate(sources, 1):
        meta = src.get("metadata", {})
        fname = meta.get("source", "Unknown")
        # Strip path prefix if present
        fname = fname.split("/")[-1].split("\\")[-1]
        page  = meta.get("page", "")
        page_label = f" · p.{page}" if page != "" else ""
        chips_html += (
            f'<span class="citation-chip">📄 {fname}{page_label}</span>'
        )
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
                f'  <div class="source-meta">'
                f'    <span>📄 {fname}</span>'
                f'    <span>Page {page}</span>'
                f'  </div>'
                f'  <div class="source-snippet">{snippet}…</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
# SIDEBAR — "Bring Your Own Key" Control Panel
# ─────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        # ── Brand ──────────────────────────────
        st.markdown(
            '<div style="padding:0.6rem 0 1rem 0;">'
            '  <span style="font-family:\'DM Serif Display\',serif;font-size:1.4rem;'
            'color:#f5a623;letter-spacing:-0.02em;">⬡ NeuralDocs</span>'
            '  <div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
            'color:#7a7a8a;text-transform:uppercase;letter-spacing:0.1em;margin-top:2px;">'
            'Enterprise RAG Platform</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── System Status ──────────────────────
        st.markdown("**System Status**")
        health = _api_health()
        if health:
            st.markdown(
                '<span class="status-pill status-ok">● API Online</span>',
                unsafe_allow_html=True,
            )
            vs_status = health.get("vector_store", "empty")
            vs_class  = "status-ok" if vs_status == "ready" else "status-warn"
            st.markdown(
                f'<span class="status-pill {vs_class}">● ChromaDB {vs_status}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<span style="font-family:DM Mono,monospace;font-size:0.68rem;'
                f'color:#7a7a8a;">{health.get("docs_indexed","0")} chunks indexed</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-pill status-err">● API Offline</span>',
                unsafe_allow_html=True,
            )

        # Key validity indicator
        if st.session_state.key_valid is True:
            st.markdown(
                '<span class="status-pill status-ok">● LLM Key Valid</span>',
                unsafe_allow_html=True,
            )
        elif st.session_state.key_valid is False:
            st.markdown(
                '<span class="status-pill status-err">● LLM Key Invalid</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-pill status-warn">● LLM Key Not Set</span>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── LLM Provider Picker ────────────────
        st.markdown("**🧠 AI Provider**")

        provider = st.selectbox(
            "Provider",
            options=list(PROVIDER_MODELS.keys()),
            index=list(PROVIDER_MODELS.keys()).index(st.session_state.provider),
            label_visibility="collapsed",
        )

        # Reset model when provider changes
        if provider != st.session_state.provider:
            st.session_state.provider = provider
            st.session_state.model    = PROVIDER_MODELS[provider][0]
            st.session_state.key_valid = None
            st.session_state.key_msg   = ""

        model = st.selectbox(
            "Model",
            options=PROVIDER_MODELS[provider],
            index=(
                PROVIDER_MODELS[provider].index(st.session_state.model)
                if st.session_state.model in PROVIDER_MODELS[provider]
                else 0
            ),
            label_visibility="collapsed",
        )
        st.session_state.model = model

        # ── API Key Input ──────────────────────
        # Security note: password=True masks the key on screen.
        # The value is stored in st.session_state["api_key"] which is:
        #   • Per browser tab / session
        #   • Never written to disk or logs
        #   • Garbage-collected when the tab closes
        if provider == "Local (Ollama)":
            st.info("🦙 Ollama runs locally — no API key needed.")
            st.session_state.api_key   = ""
            st.session_state.key_valid = True
            st.session_state.key_msg   = "Local model — no key required."
        else:
            api_key = st.text_input(
                f"{provider} API Key",
                value=st.session_state.api_key,
                type="password",       # ← renders as ••••••
                placeholder="Paste your API key…",
                help=(
                    "Your key is stored only in browser session memory. "
                    "It is never saved to disk, logged, or transmitted "
                    "beyond the local FastAPI backend."
                ),
            )
            if api_key != st.session_state.api_key:
                st.session_state.api_key   = api_key
                # Re-validate on change
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

        # ── Document Upload & Ingestion ────────
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
                            f"✅ {result['chunks_ingested']} chunks from "
                            f"{len(uploaded)} file(s)."
                        )
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")

        st.divider()

        # ── Session Controls ───────────────────
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
        st.caption(
            "Built by **Akmal Raxmatov** · "
            "[GitHub](https://github.com/thed700)"
        )


# ─────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────

def _render_header() -> None:
    """Render the top header bar with provider badge."""
    provider_html = (
        f'<span class="provider-badge">'
        f'{st.session_state.provider} / {st.session_state.model}'
        f'</span>'
    )
    st.markdown(
        f'<div class="nd-header">'
        f'  <div>'
        f'    <div class="nd-logo">⬡ NeuralDocs</div>'
        f'    <div class="nd-tagline">'
        f'      Hybrid Search · Re-ranking · {provider_html}'
        f'    </div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_message_history() -> None:
    """Replay all messages from session state into chat bubbles."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_citations(msg["sources"])


def _guard_llm_ready() -> bool:
    """
    Return True if the LLM config is valid enough to send a query.
    Shows a friendly warning in the chat area if not.
    """
    provider = st.session_state.provider
    if provider == "Local (Ollama)":
        return True

    if not st.session_state.api_key:
        st.info(
            f"👈 Paste your **{provider}** API key in the sidebar to start chatting.",
            icon="🔑",
        )
        return False

    if st.session_state.key_valid is False:
        st.warning(
            f"The API key for **{provider}** doesn't look right. "
            "Check the sidebar for details.",
            icon="⚠️",
        )
        return False

    return True


def _handle_user_input(prompt: str) -> None:
    """
    Process a new user message:
      1. Append to history and render immediately.
      2. Call the backend with session-state credentials.
      3. Stream the response word-by-word.
      4. Render citations and append result to history.
    """
    # Show user bubble right away
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        try:
            result  = _api_query(prompt)
            answer  = result.get("answer", "")
            sources = result.get("sources", [])

            # Simulate streaming for the word-by-word effect
            st.write_stream(_stream_text(answer))

            # Citations below the answer
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
            st.session_state.messages.append(
                {"role": "assistant", "content": err}
            )
        except Exception as e:
            err = f"❌ Error: {e}"
            st.error(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": err}
            )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main() -> None:
    _render_sidebar()
    _render_header()
    st.divider()

    # Welcome state (empty history)
    if not st.session_state.messages:
        st.markdown(
            '<div style="text-align:center;padding:3rem 0;color:#7a7a8a;">'
            '  <div style="font-size:2.5rem;margin-bottom:0.5rem;">⬡</div>'
            '  <div style="font-family:DM Serif Display,serif;font-size:1.2rem;'
            'color:#e8e8ec;margin-bottom:0.4rem;">Ask anything about your documents</div>'
            '  <div style="font-size:0.82rem;">'
            'Upload files via the sidebar · Select your AI provider · Start chatting'
            '  </div>'
            '</div>',
            unsafe_allow_html=True,
        )

    _render_message_history()

    # Guard: only show input when LLM config is ready
    if _guard_llm_ready():
        if prompt := st.chat_input(
            "Ask a question about your documents…",
            key="chat_input",
        ):
            _handle_user_input(prompt)


if __name__ == "__main__":
    main()
