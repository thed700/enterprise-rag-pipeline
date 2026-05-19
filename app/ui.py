
"""
ui.py — AuraRAG modern Streamlit chat UI.
"""

from __future__ import annotations

import html
import json
import os
import time
import uuid
from typing import Any, Dict, Generator, Iterable, List, Tuple

import mistune
import requests
import streamlit as st

from app.constants import (
    PROVIDER_MODELS as FALLBACK_PROVIDERS,
    friendly_model_label,
    is_ollama_provider,
    provider_model_options,
    validate_provider_config,
)

st.set_page_config(
    page_title="AuraRAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

DEFAULT_PROMPTS = {
    "rewrite": "",
    "grade": "",
    "generate": "",
    "reflect": "",
}

SUPPORTED_UPLOAD_TYPES = ["pdf", "txt", "csv", "json", "xlsx", "xls", "parquet"]

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

STYLES = """
<style>
:root{
  --bg:#0b0c10;
  --panel:#111319;
  --panel-2:#151922;
  --panel-3:#1b2030;
  --border:rgba(255,255,255,.08);
  --text:#e8ecf4;
  --muted:#97a0b3;
  --accent:#7c9cff;
  --accent-2:#9d7cff;
  --good:#29d17f;
  --warn:#ffb84d;
  --bad:#ff6b6b;
  --radius:22px;
  --shadow:0 16px 50px rgba(0,0,0,.35);
}

html, body, [class*="css"] { background: var(--bg); color: var(--text); }
.stApp { background: radial-gradient(circle at top, rgba(124,156,255,.14), transparent 32%), var(--bg); }

[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(17,19,25,.96), rgba(11,12,16,.98));
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text); }

.aura-shell {
  border: 1px solid var(--border);
  background: rgba(17,19,25,.82);
  box-shadow: var(--shadow);
  border-radius: 28px;
  padding: 18px 20px;
}

.aura-title {
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.04em;
  margin-bottom: .1rem;
}
.aura-subtitle {
  color: var(--muted);
  font-size: .92rem;
  margin-bottom: 0;
}

.aura-pill {
  display:inline-flex;
  align-items:center;
  gap:.35rem;
  padding:.32rem .7rem;
  border-radius:999px;
  border:1px solid var(--border);
  background: rgba(255,255,255,.04);
  color: var(--text);
  font-size:.78rem;
  margin:.15rem .2rem .15rem 0;
}

.aura-card {
  border:1px solid var(--border);
  border-radius: 20px;
  background: rgba(255,255,255,.03);
  box-shadow: var(--shadow);
  padding: 14px 16px;
}

.aura-muted { color: var(--muted); font-size: .88rem; }
.aura-small { color: var(--muted); font-size: .78rem; }

[data-testid="stChatMessage"] {
  border: 1px solid var(--border);
  border-radius: 24px;
  background: rgba(255,255,255,.02);
  margin-bottom: 14px;
  padding: 4px 2px 4px 2px;
}

[data-testid="stChatInput"] textarea {
  border-radius: 18px !important;
  border: 1px solid var(--border) !important;
  background: rgba(255,255,255,.04) !important;
  color: var(--text) !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: rgba(124,156,255,.85) !important;
  box-shadow: 0 0 0 3px rgba(124,156,255,.13) !important;
}

.code-shell {
  position: relative;
  margin: .7rem 0 1rem 0;
  border-radius: 18px;
  border: 1px solid var(--border);
  overflow: hidden;
  background: #0a0d14;
}
.code-toolbar {
  display:flex;
  justify-content:flex-end;
  gap:.5rem;
  padding:.55rem .65rem;
  background: rgba(255,255,255,.03);
  border-bottom: 1px solid var(--border);
}
.copy-btn {
  border: 1px solid var(--border);
  background: rgba(255,255,255,.05);
  color: var(--text);
  border-radius: 999px;
  padding: .3rem .75rem;
  font-size: .76rem;
  cursor: pointer;
}
.copy-btn:hover { border-color: rgba(124,156,255,.7); }
.code-shell pre {
  margin:0;
  padding: 1rem 1rem 1rem 1rem;
  overflow-x:auto;
  white-space: pre;
}
.code-shell code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: .86rem;
  color: #d7def7;
}

.source-card {
  border: 1px solid var(--border);
  background: rgba(255,255,255,.03);
  border-radius: 18px;
  padding: 12px 14px;
  margin: 10px 0;
}
.source-meta {
  display:flex;
  flex-wrap:wrap;
  gap: 6px;
  margin-bottom: 8px;
}
.source-chip {
  display:inline-flex;
  align-items:center;
  padding: 3px 8px;
  border-radius: 999px;
  font-size: .72rem;
  border: 1px solid var(--border);
  background: rgba(255,255,255,.04);
  color: var(--muted);
}

.stepper {
  display:grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}
.step {
  border:1px solid var(--border);
  border-radius: 16px;
  background: rgba(255,255,255,.03);
  padding: 10px 12px;
}
.step.active { border-color: rgba(124,156,255,.7); background: rgba(124,156,255,.10); }
.step.done { border-color: rgba(41,209,127,.4); background: rgba(41,209,127,.08); }
.step-title { font-weight: 600; font-size: .84rem; }
.step-sub { color: var(--muted); font-size: .75rem; margin-top: 3px; }

hr { border-color: var(--border) !important; }

#MainMenu, footer, header { visibility: hidden; }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _bootstrap_state() -> None:
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = f"session-{uuid.uuid4().hex[:8]}"
    if "stream_enabled" not in st.session_state:
        st.session_state.stream_enabled = True
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5
    if "provider" not in st.session_state:
        st.session_state.provider = "OpenAI"
    if "model" not in st.session_state:
        st.session_state.model = "gpt-4o-mini"
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = "llama3"
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "custom_model" not in st.session_state:
        st.session_state.custom_model = ""
    if "system_prompts" not in st.session_state:
        st.session_state.system_prompts = dict(DEFAULT_PROMPTS)
    if "last_health" not in st.session_state:
        st.session_state.last_health = None


def _current_session_messages() -> List[Dict[str, Any]]:
    sessions = st.session_state.sessions
    return sessions.setdefault(st.session_state.active_session_id, [])


def _safe_model_choice(provider: str, model: str) -> str:
    if provider == "Local (Ollama)":
        return model or st.session_state.ollama_model or "llama3"
    return model or "gpt-4o-mini"


def _provider_models(providers: Dict[str, List[str]], provider: str) -> List[str]:
    values = providers.get(provider) or provider_model_options(provider) or FALLBACK_PROVIDERS.get(provider, [])
    return list(values)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def _fetch_providers() -> Dict[str, List[str]]:
    try:
        resp = requests.get(f"{API_BASE}/providers", timeout=4)
        resp.raise_for_status()
        payload = resp.json()
        providers = payload.get("providers", {})
        if isinstance(providers, dict) and providers:
            return providers
    except Exception:
        pass
    return FALLBACK_PROVIDERS


@st.cache_data(ttl=15)
def _fetch_health() -> Dict[str, Any] | None:
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=4)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _upload_files(files: List[Any]) -> Dict[str, Any]:
    multipart = [("files", (file.name, file.getvalue(), file.type or "application/octet-stream")) for file in files]
    resp = requests.post(f"{API_BASE}/ingest", files=multipart, timeout=300)
    resp.raise_for_status()
    return resp.json()


def _query_once(payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{API_BASE}/query", json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()


def _query_stream(payload: Dict[str, Any]) -> Generator[Tuple[str, str], None, None]:
    with requests.post(f"{API_BASE}/query/stream", json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if not raw.startswith("data: "):
                continue
            data = raw[6:].strip()
            if data == "[DONE]":
                yield ("done", "")
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "token" in obj:
                yield ("token", str(obj["token"]))
            elif "error" in obj:
                yield ("error", str(obj["error"]))
            elif "meta" in obj:
                yield ("meta", json.dumps(obj["meta"], ensure_ascii=False))


# ---------------------------------------------------------------------------
# Markdown / code rendering
# ---------------------------------------------------------------------------

class _CopyCodeRenderer(mistune.HTMLRenderer):
    def block_code(self, code: str, info: str | None = None) -> str:
        language = (info or "").strip()
        code_text = code.rstrip("\n")
        copy_id = f"copy-{uuid.uuid4().hex}"
        safe_code = html.escape(code_text)
        button_label = "Copy"
        return f"""
<div class="code-shell">
  <div class="code-toolbar">
    <button class="copy-btn" onclick="navigator.clipboard.writeText(this.dataset.copy); this.textContent='Copied'; setTimeout(() => this.textContent='{button_label}', 1600);" data-copy="{html.escape(code_text, quote=True)}">{button_label}</button>
  </div>
  <pre><code class="language-{html.escape(language, quote=True)}" id="{copy_id}">{safe_code}</code></pre>
</div>
"""

def _markdown_renderer() -> mistune.Markdown:
    renderer = _CopyCodeRenderer(escape=True, allow_harmful_protocols=False)
    return mistune.create_markdown(renderer=renderer, plugins=["table", "strikethrough", "url"])


_MD = _markdown_renderer()


def render_markdown(text: str) -> None:
    st.markdown(_MD(text), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar(providers: Dict[str, List[str]]) -> Dict[str, Any]:
    with st.sidebar:
        st.markdown('<div class="aura-shell">', unsafe_allow_html=True)
        st.markdown('<div class="aura-title">AuraRAG</div>', unsafe_allow_html=True)
        st.markdown('<div class="aura-subtitle">Enterprise retrieval chat</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        health = _fetch_health()
        st.session_state.last_health = health

        st.markdown("### Runtime")
        if health:
            st.markdown(
                f"""
                <div class="aura-card">
                  <div class="aura-pill">API: <strong>online</strong></div>
                  <div class="aura-pill">Vector store: <strong>{health.get('vector_store', '—')}</strong></div>
                  <div class="aura-pill">BM25: <strong>{health.get('bm25_index', '—')}</strong></div>
                  <div class="aura-pill">Indexed docs: <strong>{health.get('docs_indexed', '0')}</strong></div>
                  <div class="aura-pill">Sessions: <strong>{health.get('active_sessions', '0')}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="aura-card"><span class="aura-pill">API: <strong>offline</strong></span></div>', unsafe_allow_html=True)

        st.markdown("### Session")
        session_ids = list(st.session_state.sessions.keys())
        if st.session_state.active_session_id not in session_ids:
            session_ids.insert(0, st.session_state.active_session_id)

        cols = st.columns(2)
        with cols[0]:
            if st.button("New session", use_container_width=True):
                st.session_state.active_session_id = f"session-{uuid.uuid4().hex[:8]}"
                st.session_state.sessions.setdefault(st.session_state.active_session_id, [])
                st.rerun()
        with cols[1]:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.sessions[st.session_state.active_session_id] = []
                st.rerun()

        if session_ids:
            st.session_state.active_session_id = st.selectbox(
                "Active session",
                options=session_ids,
                index=session_ids.index(st.session_state.active_session_id),
            )

        st.markdown(f'<div class="aura-small">Local sessions: {len(st.session_state.sessions)}</div>', unsafe_allow_html=True)

        st.markdown("### Ingest")
        uploads = st.file_uploader(
            "Upload documents",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        ingest_clicked = st.button("Index selected files", use_container_width=True)
        if ingest_clicked:
            if not uploads:
                st.warning("Select one or more files first.")
            else:
                with st.spinner("Indexing files..."):
                    try:
                        result = _upload_files(uploads)
                        st.success(
                            f"Indexed {result.get('chunks_ingested', 0)} chunks"
                            f" · skipped {result.get('duplicates_skipped', 0)} duplicates"
                        )
                        st.caption(result.get("message", ""))
                        _fetch_health.clear()
                    except Exception as exc:
                        st.error(f"Ingest failed: {exc}")

        st.markdown("### Retrieval settings")
        st.session_state.top_k = st.slider("Top K", min_value=1, max_value=20, value=int(st.session_state.top_k), step=1)
        st.session_state.stream_enabled = st.toggle("Stream answers", value=bool(st.session_state.stream_enabled))

        st.markdown("### Provider & model")
        provider = st.selectbox(
            "Provider",
            list(providers.keys()),
            index=list(providers.keys()).index(st.session_state.provider) if st.session_state.provider in providers else 0,
        )
        st.session_state.provider = provider

        model_options = _provider_models(providers, provider)
        if is_ollama_provider(provider):
            st.session_state.ollama_model = st.text_input(
                "Ollama model",
                value=st.session_state.ollama_model or "llama3",
                placeholder="llama3, mistral, qwen2.5, ...",
            )
            model = st.session_state.ollama_model.strip() or "llama3"
        else:
            selected = st.selectbox(
                "Model",
                options=model_options or ["gpt-4o-mini"],
                index=0,
                format_func=friendly_model_label,
            )
            st.session_state.custom_model = st.text_input(
                "Custom model override",
                value=st.session_state.custom_model,
                placeholder="Leave blank to use the dropdown selection",
            )
            model = st.session_state.custom_model.strip() or selected

        st.session_state.model = model

        if provider == "Local (Ollama)":
            st.info("No API key is required for Ollama.")
            st.session_state.api_key = ""
        else:
            st.session_state.api_key = st.text_input(
                "API key",
                value=st.session_state.api_key,
                type="password",
                placeholder="Paste your provider API key here",
            )
            valid, message = validate_provider_config(provider, st.session_state.api_key)
            st.caption(message)
            if not valid:
                st.warning(message)

        st.markdown("### Prompt overrides")
        with st.expander("Edit system prompts", expanded=False):
            st.session_state.system_prompts["rewrite"] = st.text_area(
                "Rewrite",
                value=st.session_state.system_prompts.get("rewrite", ""),
                height=110,
                placeholder="Leave blank to use the backend default rewrite prompt.",
            )
            st.session_state.system_prompts["grade"] = st.text_area(
                "Grade",
                value=st.session_state.system_prompts.get("grade", ""),
                height=110,
                placeholder="Leave blank to use the backend default grading prompt.",
            )
            st.session_state.system_prompts["generate"] = st.text_area(
                "Generate",
                value=st.session_state.system_prompts.get("generate", ""),
                height=140,
                placeholder="Leave blank to use the backend default answer prompt.",
            )
            st.session_state.system_prompts["reflect"] = st.text_area(
                "Reflect",
                value=st.session_state.system_prompts.get("reflect", ""),
                height=110,
                placeholder="Leave blank to use the backend default reflection prompt.",
            )
            if st.button("Reset prompt overrides", use_container_width=True):
                st.session_state.system_prompts = dict(DEFAULT_PROMPTS)
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    return {
        "provider": provider,
        "model": model,
    }


# ---------------------------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------------------------

def _render_pipeline_steps(current: str = "idle") -> None:
    order = ["rewrite", "retrieve", "grade", "generate"]
    labels = {
        "rewrite": "Rewrite",
        "retrieve": "Retrieve",
        "grade": "Grade",
        "generate": "Generate",
    }
    state_map = {step: "done" if order.index(step) < order.index(current) else "active" if step == current else "" for step in order if current in order}
    cols_html = []
    for step in order:
        class_name = "step"
        if current in order:
            if step == current:
                class_name += " active"
            elif order.index(step) < order.index(current):
                class_name += " done"
        cols_html.append(
            f'<div class="{class_name}"><div class="step-title">{labels[step]}</div><div class="step-sub">LangGraph node</div></div>'
        )
    st.markdown(f'<div class="stepper">{"".join(cols_html)}</div>', unsafe_allow_html=True)


def _render_source_cards(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for i, source in enumerate(sources, start=1):
            metadata = source.get("metadata", {}) or {}
            chips = []
            for key in ("source", "file_type", "sheet_name", "row_index", "page"):
                if key in metadata:
                    chips.append(f'<span class="source-chip">{html.escape(key)}: {html.escape(str(metadata[key]))}</span>')
            st.markdown(
                f"""
                <div class="source-card">
                  <div class="source-meta">
                    <span class="source-chip">#{i}</span>
                    {''.join(chips)}
                  </div>
                  <div>{html.escape(str(source.get('content', '')))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _append_message(session_id: str, role: str, content: str, sources: List[Dict[str, Any]] | None = None, meta: Dict[str, Any] | None = None) -> None:
    session = st.session_state.sessions.setdefault(session_id, [])
    session.append(
        {
            "role": role,
            "content": content,
            "sources": sources or [],
            "meta": meta or {},
        }
    )


def _render_history(session_id: str) -> None:
    history = st.session_state.sessions.get(session_id, [])
    for message in history:
        role = message["role"]
        avatar = "🧑" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            if role == "assistant":
                render_markdown(message["content"])
                if message.get("meta"):
                    meta = message["meta"]
                    pipeline_trace = meta.get("pipeline_trace", [])
                    if pipeline_trace:
                        st.caption("Pipeline: " + " → ".join(pipeline_trace))
                if message.get("sources"):
                    _render_source_cards(message["sources"])
            else:
                st.markdown(message["content"])


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    _bootstrap_state()
    providers = _fetch_providers()
    sidebar_state = _render_sidebar(providers)

    st.markdown(
        f"""
        <div class="aura-shell">
          <div class="aura-title">Chat with AuraRAG</div>
          <p class="aura-subtitle">Grounded answers from your indexed documents. Streamed responses, row-level dataset ingestion, and prompt overrides are built in.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_history(st.session_state.active_session_id)

    prompt = st.chat_input("Ask a question about your indexed documents...")
    if not prompt:
        return

    session_id = st.session_state.active_session_id
    _append_message(session_id, "user", prompt)

    payload = {
        "question": prompt,
        "top_k": int(st.session_state.top_k),
        "provider": sidebar_state["provider"],
        "model": sidebar_state["model"],
        "api_key": st.session_state.api_key,
        "session_id": session_id,
        "system_prompts": {
            key: value.strip()
            for key, value in st.session_state.system_prompts.items()
            if isinstance(value, str) and value.strip()
        },
    }

    with st.chat_message("assistant", avatar="🤖"):
        status_placeholder = st.empty()
        answer_placeholder = st.empty()
        current_step = "rewrite"
        status_placeholder.markdown(
            "<div class='aura-card'>"
            "<div class='aura-pill'>Executing LangGraph pipeline</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        _render_pipeline_steps(current_step)

        final_answer = ""
        sources: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {"pipeline_trace": []}

        try:
            if st.session_state.stream_enabled:
                first_token_seen = False
                step_announced = False
                stream_buffer = ""
                for kind, value in _query_stream(payload):
                    if kind == "token":
                        if not first_token_seen:
                            first_token_seen = True
                            current_step = "generate"
                            status_placeholder.markdown(
                                "<div class='aura-card'><span class='aura-pill'>Rewrite → Retrieve → Grade complete</span><span class='aura-pill'>Generating answer</span></div>",
                                unsafe_allow_html=True,
                            )
                            _render_pipeline_steps(current_step)
                        stream_buffer += value
                        final_answer = stream_buffer
                        answer_placeholder.markdown(_MD(stream_buffer), unsafe_allow_html=True)
                    elif kind == "meta":
                        try:
                            meta = json.loads(value)
                        except Exception:
                            meta = {"raw_meta": value}
                    elif kind == "error":
                        raise RuntimeError(value)
                    elif kind == "done":
                        break

                if not final_answer:
                    current_step = "generate"
                    _render_pipeline_steps(current_step)
                    response = _query_once(payload)
                    final_answer = response.get("answer", "")
                    sources = response.get("sources", [])
                    meta = {
                        "pipeline_trace": response.get("pipeline_trace", []),
                        "graded_chunks": response.get("graded_chunks", 0),
                        "reflect_loops": response.get("reflect_loops", 0),
                    }
                else:
                    # Fetch the final structured response so sources and trace are retained.
                    try:
                        response = _query_once(payload)
                        sources = response.get("sources", [])
                        meta.update(
                            {
                                "pipeline_trace": response.get("pipeline_trace", meta.get("pipeline_trace", [])),
                                "graded_chunks": response.get("graded_chunks", meta.get("graded_chunks", 0)),
                                "reflect_loops": response.get("reflect_loops", meta.get("reflect_loops", 0)),
                            }
                        )
                    except Exception:
                        pass
            else:
                status_placeholder.markdown(
                    "<div class='aura-card'><span class='aura-pill'>Running non-streaming query</span></div>",
                    unsafe_allow_html=True,
                )
                response = _query_once(payload)
                final_answer = response.get("answer", "")
                sources = response.get("sources", [])
                meta = {
                    "pipeline_trace": response.get("pipeline_trace", []),
                    "graded_chunks": response.get("graded_chunks", 0),
                    "reflect_loops": response.get("reflect_loops", 0),
                }

        except Exception as exc:
            final_answer = f"Error: {exc}"
            st.error(final_answer)

        if final_answer:
            answer_placeholder.markdown(_MD(final_answer), unsafe_allow_html=True)

        _render_source_cards(sources)
        if meta.get("pipeline_trace"):
            st.caption("Pipeline: " + " → ".join(meta["pipeline_trace"]))

        _append_message(session_id, "assistant", final_answer or "No answer returned.", sources=sources, meta=meta)


if __name__ == "__main__":
    main()
