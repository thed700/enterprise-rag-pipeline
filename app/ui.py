"""
ui.py — AuraRAG v3.4 — Redesigned UI
"""

from __future__ import annotations

import html
import json
import os
import uuid
from typing import Any, Dict, Generator, List, Tuple

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

DEFAULT_PROMPTS = {"rewrite": "", "grade": "", "generate": "", "reflect": ""}
SUPPORTED_UPLOAD_TYPES = ["pdf", "txt", "csv", "json", "xlsx", "xls", "parquet"]

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:ital,opsz,wght@0,6..12,300;0,6..12,400;0,6..12,500;0,6..12,600;1,6..12,300&family=Fira+Code:wght@400;500&display=swap');

:root {
  --bg:        #0c0d11;
  --surface:   #13151e;
  --surface2:  #191c27;
  --surface3:  #20243380;
  --surface4:  #252939;
  --border:    rgba(255,255,255,.07);
  --border2:   rgba(255,255,255,.12);
  --border3:   rgba(255,255,255,.18);
  --text:      #e6eaf4;
  --text2:     #a8b0c8;
  --muted:     #5a6278;
  --accent:    #6e9fff;
  --accent2:   #a47cff;
  --accent3:   #4ddec8;
  --grad:      linear-gradient(135deg, #6e9fff 0%, #a47cff 100%);
  --grad2:     linear-gradient(135deg, #4ddec8 0%, #6e9fff 100%);
  --good:      #34d48a;
  --warn:      #f5a524;
  --bad:       #ff6060;
  --r-xs:      8px;
  --r-sm:      12px;
  --r-md:      16px;
  --r-lg:      22px;
  --r-xl:      28px;
  --r-2xl:     36px;
  --r-pill:    999px;
  --shadow:    0 4px 24px rgba(0,0,0,.55);
  --shadow2:   0 8px 40px rgba(0,0,0,.65);
  --font:      'Nunito Sans', system-ui, sans-serif;
  --mono:      'Fira Code', monospace;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
  font-family: var(--font) !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}

.stApp {
  background:
    radial-gradient(ellipse 60% 35% at 20% -5%,  rgba(124,158,255,.09), transparent),
    radial-gradient(ellipse 50% 40% at 85%  5%,  rgba(168,124,255,.07), transparent),
    radial-gradient(ellipse 40% 30% at 50% 100%, rgba(92,232,192,.05),  transparent),
    var(--bg) !important;
  font-family: var(--font) !important;
}

/* ─── SIDEBAR ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(170deg,
    rgba(19,21,28,.99) 0%,
    rgba(13,15,20,.995) 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
  color: var(--text) !important;
  font-family: var(--font) !important;
}

/* Brand block */
.sb-brand {
  padding: 6px 0 22px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 6px;
}
.sb-logo-row {
  display: flex; align-items: center; gap: 11px; margin-bottom: 0;
}
.sb-logo {
  width: 38px; height: 38px;
  border-radius: 11px;
  background: var(--grad);
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; flex-shrink: 0;
  box-shadow: 0 3px 14px rgba(124,158,255,.35);
  letter-spacing: 0;
}
.sb-name { font-size: 1.1rem; font-weight: 600; letter-spacing: -.03em; }
.sb-tagline {
  font-size: .72rem; color: var(--muted) !important;
  margin-top: 4px; letter-spacing: .01em;
}

/* section labels */
.sb-label {
  font-size: .63rem;
  font-weight: 600;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--muted) !important;
  margin: 18px 0 7px;
  padding-left: 2px;
}

/* status pill */
.sb-status-row {
  display: flex; align-items: center; gap: 9px;
  padding: 9px 13px;
  border-radius: var(--r-md);
  border: 1px solid var(--border);
  background: var(--surface);
  font-size: .82rem;
  margin-bottom: 7px;
}
.sb-dot {
  width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}
.sb-dot.on  {
  background: var(--good);
  box-shadow: 0 0 0 3px rgba(61,214,140,.18);
  animation: pulse-on 2.5s infinite;
}
.sb-dot.off { background: var(--bad); box-shadow: 0 0 6px var(--bad); }
@keyframes pulse-on {
  0%,100% { box-shadow: 0 0 0 3px rgba(61,214,140,.18); }
  50%      { box-shadow: 0 0 0 5px rgba(61,214,140,.08); }
}

/* stat chips below status */
.sb-stats {
  display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 2px;
}
.sb-stat {
  font-size: .71rem; padding: 3px 9px;
  border-radius: var(--r-pill);
  border: 1px solid var(--border);
  background: var(--surface2);
  color: var(--text2) !important;
}
.sb-stat strong { color: var(--text) !important; font-weight: 500; }

/* sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
  border-radius: var(--r-sm) !important;
  border: 1px solid var(--border2) !important;
  background: var(--surface2) !important;
  color: var(--text2) !important;
  font-family: var(--font) !important;
  font-size: .81rem !important;
  font-weight: 500 !important;
  height: 34px !important;
  transition: all .15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: var(--surface3) !important;
  border-color: rgba(124,158,255,.35) !important;
  color: var(--text) !important;
}

/* sidebar selectbox / text inputs */
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--r-sm) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
  font-size: .84rem !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div:focus-within,
[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
  border-color: rgba(124,158,255,.45) !important;
  box-shadow: 0 0 0 3px rgba(124,158,255,.08) !important;
}

/* slider */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
  color: var(--accent) !important;
}

/* toggle */
[data-testid="stSidebar"] input[type="checkbox"]:checked + div {
  background: var(--accent) !important;
}

/* caption */
.stCaption, [data-testid="stCaptionContainer"] {
  color: var(--muted) !important;
  font-size: .74rem !important;
  font-family: var(--font) !important;
}

/* ─── GEMINI-STYLE CHAT INPUT ─────────────────────────────────────────── */
/* Bottom strip */
[data-testid="stBottom"] {
  background: linear-gradient(to top, var(--bg) 60%, transparent) !important;
  padding: 0 0 20px !important;
}

/* Outer pill wrapper */
[data-testid="stChatInput"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}
[data-testid="stChatInput"] > div {
  background: var(--surface) !important;
  border: 1.5px solid var(--border3) !important;
  border-radius: var(--r-2xl) !important;
  box-shadow: 0 2px 20px rgba(0,0,0,.4), 0 0 0 0 rgba(110,159,255,0) !important;
  transition: border-color .2s ease, box-shadow .2s ease !important;
  padding: 8px 8px 8px 22px !important;
  min-height: 60px !important;
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
}
[data-testid="stChatInput"] > div:focus-within {
  border-color: rgba(110,159,255,.5) !important;
  box-shadow: 0 2px 20px rgba(0,0,0,.4), 0 0 0 4px rgba(110,159,255,.10) !important;
}

/* Textarea */
[data-testid="stChatInput"] textarea {
  font-family: var(--font) !important;
  font-size: 1rem !important;
  font-weight: 400 !important;
  color: var(--text) !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
  padding: 6px 0 !important;
  min-height: 40px !important;
  max-height: 180px !important;
  resize: none !important;
  line-height: 1.55 !important;
  letter-spacing: .01em !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: var(--muted) !important;
  font-size: .94rem !important;
  font-weight: 300 !important;
}

/* Send button — glowing gradient circle */
[data-testid="stChatInput"] button {
  background: var(--grad) !important;
  border: none !important;
  border-radius: 50% !important;
  width: 42px !important;
  height: 42px !important;
  min-width: 42px !important;
  min-height: 42px !important;
  margin: 0 2px 0 0 !important;
  padding: 0 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  cursor: pointer !important;
  box-shadow: 0 2px 12px rgba(110,159,255,.45) !important;
  transition: transform .15s cubic-bezier(.34,1.56,.64,1), box-shadow .15s !important;
  flex-shrink: 0 !important;
}
[data-testid="stChatInput"] button:hover {
  transform: scale(1.08) !important;
  box-shadow: 0 4px 20px rgba(110,159,255,.6) !important;
}
[data-testid="stChatInput"] button:active { transform: scale(0.94) !important; }
[data-testid="stChatInput"] button svg {
  color: #fff !important; stroke: #fff !important;
  fill: #fff !important; width: 18px !important; height: 18px !important;
}

/* ─── CHAT MESSAGES ───────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 2px 0 !important;
  margin-bottom: 8px !important;
}

/* Both bubbles: the inner content div gets styled */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] .stMarkdown {
  font-size: .95rem !important;
  line-height: 1.65 !important;
}

/* User message — right-aligned pill style */
[data-testid="stChatMessage"]:nth-child(odd) > div > div:last-child {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--r-xl) var(--r-xl) var(--r-sm) var(--r-xl) !important;
  padding: 11px 18px !important;
  max-width: 88% !important;
}

/* Assistant avatar ring */
[data-testid="chatAvatarIcon-assistant"] {
  background: var(--grad) !important;
  border-radius: 50% !important;
  box-shadow: 0 2px 10px rgba(110,159,255,.35) !important;
}
[data-testid="chatAvatarIcon-user"] {
  background: var(--surface3) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 50% !important;
}

/* ─── MAIN AREA HEADER ────────────────────────────────────────────────── */
.main-hdr {
  display: flex; align-items: center; gap: 14px;
  padding: 20px 0 16px;
}
.main-hdr-icon {
  width: 44px; height: 44px;
  border-radius: 13px;
  background: var(--grad);
  display: flex; align-items: center; justify-content: center;
  font-size: 20px; flex-shrink: 0;
  box-shadow: 0 4px 16px rgba(124,158,255,.38);
}
.main-hdr-title {
  font-size: 1.55rem;
  font-weight: 600;
  letter-spacing: -.04em;
  line-height: 1.1;
}
.main-hdr-sub {
  font-size: .82rem;
  color: var(--muted);
  margin-top: 3px;
  font-weight: 400;
  letter-spacing: .005em;
}

/* ─── OFFLINE BANNER ─────────────────────────────────────────────────── */
.offline-banner {
  display: flex; align-items: center; gap: 10px;
  padding: 11px 16px;
  border-radius: var(--r-md);
  background: rgba(255,107,107,.07);
  border: 1px solid rgba(255,107,107,.2);
  font-size: .84rem; color: #ff9e9e;
  margin-bottom: 14px;
}
.offline-banner .blink {
  animation: blink-bad 1.2s infinite;
}
@keyframes blink-bad {
  0%,100% { opacity:1; } 50% { opacity:.3; }
}

/* ─── EMPTY STATE ────────────────────────────────────────────────────── */
.empty-wrap {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; padding: 64px 20px; gap: 14px; text-align: center;
}
.empty-logo {
  width: 64px; height: 64px; border-radius: 20px;
  background: var(--grad);
  display: flex; align-items: center; justify-content: center;
  font-size: 28px;
  box-shadow: 0 6px 30px rgba(124,158,255,.3);
  margin-bottom: 4px;
}
.empty-greeting {
  font-size: 1.8rem; font-weight: 600;
  letter-spacing: -.04em; line-height: 1.1;
  background: var(--grad); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; background-clip: text;
}
.empty-sub {
  font-size: .88rem; color: var(--muted);
  max-width: 340px; line-height: 1.6;
}
/* Suggestion chips */
.suggestion-row {
  display: flex; flex-wrap: wrap; justify-content: center;
  gap: 8px; margin-top: 6px; max-width: 560px;
}
.sg-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 8px 14px;
  border-radius: var(--r-pill);
  border: 1px solid var(--border2);
  background: var(--surface);
  color: var(--text2);
  font-size: .81rem; font-weight: 400;
  cursor: pointer;
  transition: all .18s;
  font-family: var(--font);
}
.sg-chip:hover {
  border-color: rgba(124,158,255,.45);
  background: var(--surface2);
  color: var(--text);
}

/* ─── THINKING / STATUS INDICATOR ───────────────────────────────────── */
.thinking-wrap {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 8px 15px;
  border-radius: var(--r-pill);
  background: var(--surface2);
  border: 1px solid var(--border);
  font-size: .82rem; color: var(--text2);
  margin: 4px 0 8px;
}
.thinking-dots { display:inline-flex; gap:3px; align-items:center; }
.thinking-dots span {
  display: inline-block;
  width: 5px; height: 5px;
  border-radius: 50%;
  background: var(--accent);
  animation: tdot 1.5s infinite ease-in-out;
}
.thinking-dots span:nth-child(2) { animation-delay: .18s; }
.thinking-dots span:nth-child(3) { animation-delay: .36s; }
@keyframes tdot {
  0%,80%,100% { transform: scale(.55); opacity:.35; }
  40%          { transform: scale(1);   opacity:1; }
}

/* ─── PIPELINE STEPPER ───────────────────────────────────────────────── */
.stepper {
  display: grid; grid-template-columns: repeat(4,1fr);
  gap: 7px; margin: 8px 0 12px;
}
.step {
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  background: var(--surface2);
  padding: 9px 12px;
  transition: all .22s;
}
.step.active {
  border-color: rgba(124,158,255,.55);
  background: rgba(124,158,255,.07);
  box-shadow: 0 0 0 3px rgba(124,158,255,.07);
}
.step.done {
  border-color: rgba(61,214,140,.35);
  background: rgba(61,214,140,.05);
}
.step-icon { font-size: .95rem; margin-bottom: 3px; }
.step-title { font-weight: 500; font-size: .79rem; color: var(--text); }
.step-sub   { color: var(--muted); font-size: .69rem; margin-top: 2px; }
.step.active .step-title { color: var(--accent); }
.step.done   .step-icon  { color: var(--good); }

/* ─── SOURCE CARDS ───────────────────────────────────────────────────── */
.source-card {
  border: 1px solid var(--border);
  background: var(--surface2);
  border-radius: var(--r-md);
  padding: 11px 13px;
  margin: 7px 0;
  transition: border-color .2s;
}
.source-card:hover { border-color: var(--border2); }
.source-meta { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 7px; }
.source-chip {
  display: inline-flex; align-items: center;
  padding: 2px 8px;
  border-radius: var(--r-pill);
  font-size: .67rem;
  border: 1px solid var(--border);
  background: var(--surface3);
  color: var(--muted);
  font-family: var(--mono);
}
.source-text {
  font-size: .81rem; color: var(--text2); line-height: 1.55;
}

/* ─── CODE BLOCKS ───────────────────────────────────────────────────── */
.code-shell {
  position: relative; margin: .55rem 0 .85rem;
  border-radius: var(--r-md);
  border: 1px solid var(--border);
  overflow: hidden; background: #070a12;
}
.code-toolbar {
  display: flex; justify-content: flex-end; gap: .35rem;
  padding: .4rem .55rem;
  background: rgba(255,255,255,.025);
  border-bottom: 1px solid var(--border);
}
.copy-btn {
  border: 1px solid var(--border);
  background: rgba(255,255,255,.045);
  color: var(--text2); border-radius: var(--r-pill);
  padding: .22rem .65rem; font-size: .71rem; cursor: pointer;
  font-family: var(--font); transition: all .14s;
}
.copy-btn:hover { border-color: rgba(124,158,255,.5); color: var(--accent); }
.code-shell pre {
  margin: 0; padding: .85rem 1rem; overflow-x: auto; white-space: pre;
}
.code-shell code {
  font-family: var(--mono); font-size: .82rem; color: #c5d0f0;
}

/* ─── MISC ───────────────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; }
#MainMenu, footer, header { visibility: hidden; }
.stMarkdown p { line-height: 1.65 !important; }
.element-container:has(.stChatMessage) { padding: 0 !important; }

/* Expander */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  background: var(--surface) !important;
}
[data-testid="stExpander"] summary {
  font-size: .83rem !important;
  font-family: var(--font) !important;
  color: var(--text2) !important;
}

/* warning / info callouts in sidebar */
[data-testid="stSidebar"] .stAlert {
  border-radius: var(--r-sm) !important;
  font-size: .8rem !important;
  border: 1px solid var(--border) !important;
  background: var(--surface2) !important;
}

/* file uploader */
[data-testid="stFileUploader"] {
  border-radius: var(--r-md) !important;
}
[data-testid="stFileUploader"] > div {
  border: 1.5px dashed var(--border2) !important;
  border-radius: var(--r-md) !important;
  background: var(--surface) !important;
  transition: border-color .2s !important;
}
[data-testid="stFileUploader"] > div:hover {
  border-color: rgba(124,158,255,.4) !important;
}
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# State bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_state() -> None:
    defs: Dict[str, Any] = {
        "sessions":          {},
        "active_session_id": f"session-{uuid.uuid4().hex[:8]}",
        "stream_enabled":    True,
        "top_k":             5,
        "provider":          "Google Gemini",
        "model":             "gemini-2.5-flash",
        "ollama_model":      "llama3",
        "api_key":           "",
        "custom_model":      "",
        "system_prompts":    dict(DEFAULT_PROMPTS),
        "last_health":       None,
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _provider_models(providers: Dict[str, List[str]], provider: str) -> List[str]:
    return list(
        providers.get(provider)
        or provider_model_options(provider)
        or FALLBACK_PROVIDERS.get(provider, [])
    )


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def _fetch_providers() -> Dict[str, List[str]]:
    try:
        resp = requests.get(f"{API_BASE}/providers", timeout=4)
        resp.raise_for_status()
        p = resp.json().get("providers", {})
        if isinstance(p, dict) and p:
            return p
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
    multipart = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
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
            if not raw or not raw.startswith("data: "):
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
# Markdown renderer
# ---------------------------------------------------------------------------

class _CopyCodeRenderer(mistune.HTMLRenderer):
    def block_code(self, code: str, info: str | None = None) -> str:
        lang = (info or "").strip()
        safe = html.escape(code.rstrip("\n"))
        copy_id = f"c{uuid.uuid4().hex[:8]}"
        safe_copy = html.escape(code.rstrip("\n"), quote=True)
        return (
            f'<div class="code-shell">'
            f'<div class="code-toolbar">'
            f'<button class="copy-btn" '
            f'onclick="navigator.clipboard.writeText(this.dataset.copy);'
            f'this.textContent=\'Copied ✓\';setTimeout(()=>this.textContent=\'Copy\',1600);" '
            f'data-copy="{safe_copy}">Copy</button>'
            f'</div>'
            f'<pre><code class="language-{html.escape(lang, quote=True)}" id="{copy_id}">{safe}</code></pre>'
            f'</div>'
        )


_MD = mistune.create_markdown(
    renderer=_CopyCodeRenderer(escape=True, allow_harmful_protocols=False),
    plugins=["table", "strikethrough", "url"],
)


def render_markdown(text: str) -> None:
    st.markdown(_MD(text), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pipeline stepper
# ---------------------------------------------------------------------------

_STEP_META = {
    "rewrite":  ("✦", "Rewrite",  "Refining query"),
    "retrieve": ("⊞", "Retrieve", "Hybrid search"),
    "grade":    ("◎", "Grade",    "Scoring chunks"),
    "generate": ("◈", "Generate", "Composing answer"),
}
_STEP_ORDER = ["rewrite", "retrieve", "grade", "generate"]


def _render_pipeline_steps(current: str) -> None:
    cur_idx = _STEP_ORDER.index(current) if current in _STEP_ORDER else -1
    parts = []
    for i, key in enumerate(_STEP_ORDER):
        icon, title, sub = _STEP_META[key]
        cls = "step"
        if i < cur_idx:
            cls += " done"
            icon = "✓"
        elif i == cur_idx:
            cls += " active"
        parts.append(
            f'<div class="{cls}">'
            f'<div class="step-icon">{icon}</div>'
            f'<div class="step-title">{title}</div>'
            f'<div class="step-sub">{sub}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Source cards
# ---------------------------------------------------------------------------

def _render_source_cards(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        return
    label = f"📎 {len(sources)} source{'s' if len(sources) != 1 else ''}"
    with st.expander(label, expanded=False):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {}) or {}
            chips = [f'<span class="source-chip">#{i}</span>']
            for key in ("source", "file_type", "sheet_name", "row_index", "page"):
                if key in meta:
                    chips.append(
                        f'<span class="source-chip">'
                        f'{html.escape(key)}: {html.escape(str(meta[key]))}'
                        f'</span>'
                    )
            content = html.escape(str(src.get("content", "")))
            st.markdown(
                f'<div class="source-card">'
                f'<div class="source-meta">{"".join(chips)}</div>'
                f'<div class="source-text">{content}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Chat history renderer
# ---------------------------------------------------------------------------

def _render_history(session_id: str) -> None:
    history = st.session_state.sessions.get(session_id, [])
    if not history:
        # Gemini-style empty / greeting state — personalised greeting
        provider = st.session_state.get("provider", "")
        model    = st.session_state.get("model", "")
        badge = ""
        if model:
            badge = (
                f'<span style="display:inline-flex;align-items:center;gap:5px;'
                f'padding:4px 12px;border-radius:999px;border:1px solid var(--border2);'
                f'background:var(--surface);font-size:.75rem;color:var(--text2);'
                f'margin-top:10px;font-weight:500;">'
                f'<span style="width:6px;height:6px;border-radius:50%;'
                f'background:var(--grad);display:inline-block;flex-shrink:0;"></span>'
                f'{html.escape(model)}</span>'
            )
        st.markdown(
            '<div class="empty-wrap">'
            '<div class="empty-logo">◈</div>'
            '<div class="empty-greeting">What can I help with?</div>'
            '<div class="empty-sub">'
            'Upload your documents in the sidebar, then ask anything — '
            'AuraRAG retrieves, grades and synthesises grounded answers.'
            '</div>'
            + badge +
            '<div class="suggestion-row">'
            '<div class="sg-chip">📄 Summarise a document</div>'
            '<div class="sg-chip">🔍 Find specific information</div>'
            '<div class="sg-chip">🧠 Compare two sections</div>'
            '<div class="sg-chip">📊 Ask about data in a table</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    for msg in history:
        role   = msg["role"]
        avatar = "🧑" if role == "user" else "◈"
        with st.chat_message(role, avatar=avatar):
            if role == "assistant":
                render_markdown(msg["content"])
                msg_meta = msg.get("meta", {})
                trace = msg_meta.get("pipeline_trace", [])
                if trace:
                    st.caption("Pipeline · " + " → ".join(trace))
                if msg.get("sources"):
                    _render_source_cards(msg["sources"])
            else:
                st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar(providers: Dict[str, List[str]]) -> Dict[str, Any]:
    with st.sidebar:

        # ── Brand ──────────────────────────────────────────────────────────
        st.markdown(
            '<div class="sb-brand">'
            '<div class="sb-logo-row">'
            '<div class="sb-logo">◈</div>'
            '<div>'
            '<div class="sb-name">AuraRAG</div>'
            '</div>'
            '</div>'
            '<div class="sb-tagline">Enterprise document intelligence</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Active model quick-badge ────────────────────────────────────────
        cur_model = st.session_state.get("model", "")
        cur_prov  = st.session_state.get("provider", "")
        if cur_model:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:7px;'
                f'padding:7px 12px;border-radius:var(--r-sm);border:1px solid var(--border);'
                f'background:var(--surface);margin-bottom:4px;">'
                f'<span style="width:6px;height:6px;border-radius:50%;'
                f'background:var(--grad);display:inline-block;flex-shrink:0;"></span>'
                f'<span style="font-size:.78rem;color:var(--text2);flex:1;min-width:0;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                f'{html.escape(cur_prov)} · {html.escape(cur_model)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Runtime status ─────────────────────────────────────────────────
        st.markdown('<div class="sb-label">Runtime</div>', unsafe_allow_html=True)
        health = _fetch_health()
        st.session_state.last_health = health

        if health:
            docs  = health.get("docs_indexed", "0")
            bm25  = health.get("bm25_index", "—")
            sess  = health.get("active_sessions", "0")
            st.markdown(
                '<div class="sb-status-row">'
                '<span class="sb-dot on"></span>'
                '<span style="font-size:.83rem">API&nbsp;<strong>online</strong></span>'
                '</div>'
                '<div class="sb-stats">'
                f'<span class="sb-stat">Docs&nbsp;<strong>{docs}</strong></span>'
                f'<span class="sb-stat">BM25&nbsp;<strong>{bm25}</strong></span>'
                f'<span class="sb-stat">Sessions&nbsp;<strong>{sess}</strong></span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="sb-status-row">'
                '<span class="sb-dot off"></span>'
                '<span style="font-size:.83rem">API&nbsp;<strong>offline</strong></span>'
                '</div>',
                unsafe_allow_html=True,
            )

        # ── Session ────────────────────────────────────────────────────────
        st.markdown('<div class="sb-label">Session</div>', unsafe_allow_html=True)

        session_ids = list(st.session_state.sessions.keys())
        if st.session_state.active_session_id not in session_ids:
            session_ids.insert(0, st.session_state.active_session_id)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("＋ New", use_container_width=True, key="btn_new_session"):
                st.session_state.active_session_id = f"session-{uuid.uuid4().hex[:8]}"
                st.session_state.sessions.setdefault(st.session_state.active_session_id, [])
                st.rerun()
        with c2:
            if st.button("✕ Clear", use_container_width=True, key="btn_clear_chat"):
                st.session_state.sessions[st.session_state.active_session_id] = []
                st.rerun()

        if len(session_ids) > 1:
            st.session_state.active_session_id = st.selectbox(
                "Active session",
                options=session_ids,
                index=session_ids.index(st.session_state.active_session_id),
                label_visibility="collapsed",
                key="session_select",
            )
        st.caption(f"{len(st.session_state.sessions)} local session(s)")

        # ── Documents ──────────────────────────────────────────────────────
        st.markdown('<div class="sb-label">Documents</div>', unsafe_allow_html=True)
        uploads = st.file_uploader(
            "Upload documents",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="file_uploader",
        )
        if st.button("⬆ Index files", use_container_width=True, key="btn_index"):
            if not uploads:
                st.warning("Select one or more files first.")
            else:
                with st.spinner("Indexing…"):
                    try:
                        result = _upload_files(uploads)
                        st.success(
                            f"✓ {result.get('chunks_ingested', 0)} chunks indexed"
                            + (f" · {result.get('duplicates_skipped', 0)} skipped" if result.get('duplicates_skipped') else "")
                        )
                        _fetch_health.clear()
                    except Exception as exc:
                        st.error(f"Ingest failed: {exc}")

        # ── Retrieval ──────────────────────────────────────────────────────
        st.markdown('<div class="sb-label">Retrieval</div>', unsafe_allow_html=True)
        st.session_state.top_k = st.slider(
            "Top K", 1, 20, int(st.session_state.top_k), 1, key="slider_topk"
        )
        st.session_state.stream_enabled = st.toggle(
            "Stream answers",
            value=bool(st.session_state.stream_enabled),
            key="toggle_stream",
        )

        # ── Model ──────────────────────────────────────────────────────────
        st.markdown('<div class="sb-label">Model</div>', unsafe_allow_html=True)

        provider_keys = list(providers.keys())
        prov_idx = provider_keys.index(st.session_state.provider) if st.session_state.provider in provider_keys else 0
        provider = st.selectbox(
            "Provider",
            provider_keys,
            index=prov_idx,
            label_visibility="collapsed",
            key="select_provider",
        )
        st.session_state.provider = provider

        model_options = _provider_models(providers, provider)

        if is_ollama_provider(provider):
            st.session_state.ollama_model = st.text_input(
                "Ollama model",
                value=st.session_state.ollama_model or "llama3",
                placeholder="llama3, mistral, qwen2.5…",
                label_visibility="collapsed",
                key="input_ollama_model",
            )
            model = st.session_state.ollama_model.strip() or "llama3"
            st.caption("No API key required for Ollama.")
            st.session_state.api_key = ""
        else:
            selected = st.selectbox(
                "Model",
                options=model_options or ["gpt-4o-mini"],
                index=0,
                format_func=friendly_model_label,
                label_visibility="collapsed",
                key="select_model",
            )
            st.session_state.custom_model = st.text_input(
                "Custom model override",
                value=st.session_state.custom_model,
                placeholder="Leave blank to use the dropdown",
                label_visibility="collapsed",
                key="input_custom_model",
            )
            model = st.session_state.custom_model.strip() or selected

            st.session_state.api_key = st.text_input(
                "API key",
                value=st.session_state.api_key,
                type="password",
                placeholder="Paste your API key…",
                label_visibility="collapsed",
                key="input_api_key",
            )
            if st.session_state.api_key:
                valid, msg = validate_provider_config(provider, st.session_state.api_key)
                st.caption(msg)

        st.session_state.model = model

        # ── Prompt overrides ───────────────────────────────────────────────
        st.markdown('<div class="sb-label">Prompt overrides</div>', unsafe_allow_html=True)
        with st.expander("Edit system prompts", expanded=False):
            for key, label, ph in [
                ("rewrite",  "Rewrite",  "Leave blank for default rewrite prompt…"),
                ("grade",    "Grade",    "Leave blank for default grading prompt…"),
                ("generate", "Generate", "Leave blank for default answer prompt…"),
                ("reflect",  "Reflect",  "Leave blank for default reflection prompt…"),
            ]:
                st.session_state.system_prompts[key] = st.text_area(
                    label,
                    value=st.session_state.system_prompts.get(key, ""),
                    height=80,
                    placeholder=ph,
                    key=f"prompt_{key}",
                )
            if st.button("Reset prompts", use_container_width=True, key="btn_reset_prompts"):
                st.session_state.system_prompts = dict(DEFAULT_PROMPTS)
                st.rerun()

    return {"provider": provider, "model": model}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _bootstrap_state()
    providers  = _fetch_providers()
    sidebar    = _render_sidebar(providers)
    health     = st.session_state.last_health

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="main-hdr">'
        '<div class="main-hdr-icon">◈</div>'
        '<div>'
        '<div class="main-hdr-title">Chat with AuraRAG</div>'
        '<div class="main-hdr-sub">Grounded answers · LangGraph agentic pipeline · Streamed responses</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Offline banner
    if not health:
        st.markdown(
            '<div class="offline-banner">'
            '<span class="blink">⚠</span>'
            '&nbsp;Backend API is offline — start the FastAPI server and reload.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Chat history ──────────────────────────────────────────────────────
    _render_history(st.session_state.active_session_id)

    # ── Input ─────────────────────────────────────────────────────────────
    prompt = st.chat_input("Ask anything about your documents…", key="chat_input")
    if not prompt:
        return

    session_id = st.session_state.active_session_id
    # BUG-AK fix: use setdefault to avoid KeyError on fresh session
    st.session_state.sessions.setdefault(session_id, []).append(
        {"role": "user", "content": prompt, "sources": [], "meta": {}}
    )

    payload: Dict[str, Any] = {
        "question":       prompt,
        "top_k":          int(st.session_state.top_k),
        "provider":       sidebar["provider"],
        "model":          sidebar["model"],
        "api_key":        st.session_state.api_key,
        "session_id":     session_id,
        "system_prompts": {
            k: v.strip()
            for k, v in st.session_state.system_prompts.items()
            if isinstance(v, str) and v.strip()
        },
    }

    with st.chat_message("assistant", avatar="◈"):
        status_ph = st.empty()
        steps_ph  = st.empty()
        answer_ph = st.empty()

        # Thinking indicator
        status_ph.markdown(
            '<div class="thinking-wrap">'
            '<div class="thinking-dots"><span></span><span></span><span></span></div>'
            '&nbsp;Thinking…'
            '</div>',
            unsafe_allow_html=True,
        )
        with steps_ph.container():
            _render_pipeline_steps("rewrite")

        final_answer = ""
        sources: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {"pipeline_trace": []}

        try:
            if st.session_state.stream_enabled:
                stream_buffer = ""
                first_token   = False

                for kind, value in _query_stream(payload):
                    if kind == "token":
                        if not first_token:
                            first_token = True
                            status_ph.markdown(
                                '<div class="thinking-wrap">'
                                '<div class="thinking-dots"><span></span><span></span><span></span></div>'
                                '&nbsp;Generating…'
                                '</div>',
                                unsafe_allow_html=True,
                            )
                            with steps_ph.container():
                                _render_pipeline_steps("generate")
                        stream_buffer += value
                        final_answer   = stream_buffer
                        answer_ph.markdown(_MD(stream_buffer), unsafe_allow_html=True)

                    elif kind == "meta":
                        try:
                            parsed_meta = json.loads(value)
                            meta = parsed_meta
                            # Bug fix: sources are embedded in the meta SSE
                            # event; extract them here so source cards render.
                            if "sources" in parsed_meta:
                                sources = parsed_meta["sources"]
                        except Exception:
                            meta = {"raw_meta": value}

                    elif kind == "error":
                        raise RuntimeError(value)

                    elif kind == "done":
                        break

                # BUG-AJ fix: only fall back when streaming produced nothing.
                # Never fire a second request after a successful stream.
                if not final_answer:
                    with steps_ph.container():
                        _render_pipeline_steps("generate")
                    response     = _query_once(payload)
                    final_answer = response.get("answer", "")
                    sources      = response.get("sources", [])
                    meta = {
                        "pipeline_trace": response.get("pipeline_trace", []),
                        "graded_chunks":  response.get("graded_chunks", 0),
                        "reflect_loops":  response.get("reflect_loops", 0),
                    }

            else:
                status_ph.markdown(
                    '<div class="thinking-wrap">'
                    '<div class="thinking-dots"><span></span><span></span><span></span></div>'
                    '&nbsp;Processing…'
                    '</div>',
                    unsafe_allow_html=True,
                )
                response     = _query_once(payload)
                final_answer = response.get("answer", "")
                sources      = response.get("sources", [])
                meta = {
                    "pipeline_trace": response.get("pipeline_trace", []),
                    "graded_chunks":  response.get("graded_chunks", 0),
                    "reflect_loops":  response.get("reflect_loops", 0),
                }

        except Exception as exc:
            final_answer = f"Something went wrong — {exc}"
            st.error(final_answer)

        # Clear status + stepper once done
        status_ph.empty()
        steps_ph.empty()

        if final_answer:
            answer_ph.markdown(_MD(final_answer), unsafe_allow_html=True)

        _render_source_cards(sources)

        trace = meta.get("pipeline_trace", [])
        if trace:
            st.caption("Pipeline · " + " → ".join(trace))

        # Persist assistant turn to session history
        st.session_state.sessions.setdefault(session_id, []).append({
            "role":    "assistant",
            "content": final_answer or "No answer returned.",
            "sources": sources,
            "meta":    meta,
        })


if __name__ == "__main__":
    main()
