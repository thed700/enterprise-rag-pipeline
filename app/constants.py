"""
constants.py — AuraRAG v4.0
Author: Akmal Raxmatov (github: thed700)

Changes v4.0:
  FEAT-MODELS: Provider registry updated to include all requested models:
    - Google Gemini: gemini-2.5-flash, gemini-2.5-pro, gemini-1.5-flash,
                     gemini-1.5-pro, gemini-2.0-flash (already in v3.4)
    - Anthropic:     claude-3-5-sonnet, claude-3-haiku (friendly aliases
                     resolving to current API identifiers)
    - OpenAI:        gpt-4o, gpt-4o-mini, gpt-4-turbo, o1-mini, o1-preview
    - Ollama:        sentinel "custom" entry; model name is supplied by the UI
                     text input and not restricted to a fixed list.
  FEAT-KEYCHK: KEY_PREFIXES updated for Google Gemini (AIza prefix).
  REFACTOR:    MODEL_LABELS extended with all date-stamped Anthropic IDs.

Shared between the backend (engine, pipeline) and the Streamlit frontend.
Importing this module has zero heavy ML dependency side-effects — safe to
import from the UI container without torch / sentence-transformers.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ─────────────────────────────────────────────
# PROVIDER → MODEL REGISTRY
# ─────────────────────────────────────────────
# Maps each provider label to the list of model IDs sent to the API.
# The Ollama entry uses a single sentinel value; the actual model name
# is typed by the user in the UI text input and replaces this at runtime.

PROVIDER_MODELS: Dict[str, List[str]] = {
    "OpenAI": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "o1-mini",
        "o1-preview",
    ],
    "Anthropic": [
        # v4.6 family — correct API identifiers
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        # Stable dated IDs
        "claude-3-5-sonnet-20241022",   # friendly label: claude-3-5-sonnet
        "claude-3-haiku-20240307",       # friendly label: claude-3-haiku
        "claude-haiku-4-5-20251001",     # friendly label: claude-haiku-4-5
        "claude-3-opus-20240229",        # friendly label: claude-3-opus
    ],
    "Google Gemini": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
    # Ollama: the sentinel "llama3" is replaced by the user's text input at
    # runtime (see ui.py _render_sidebar → Ollama branch).  The registry entry
    # exists only to satisfy provider discovery; it is never passed to the API.
    "Local (Ollama)": [
        "llama3",
        "llama3:8b",
        "llama3:70b",
        "mistral",
        "mixtral",
        "phi3",
        "phi3:mini",
        "gemma2",
        "deepseek-r1",
        "qwen2.5",
    ],
}

# ─────────────────────────────────────────────
# HUMAN-READABLE MODEL LABELS
# ─────────────────────────────────────────────
# Maps the full API model ID to the label shown in the UI dropdown.
# If a model ID is not in this dict, the ID itself is displayed as-is.

MODEL_LABELS: Dict[str, str] = {
    # Anthropic dated IDs → friendly names
    "claude-3-5-sonnet-20241022": "claude-3.5-sonnet",
    "claude-3-haiku-20240307":    "claude-3-haiku",
    "claude-haiku-4-5-20251001":  "claude-haiku-4-5",
    "claude-3-opus-20240229":     "claude-3-opus",
    # OpenAI o-series
    "o1-mini":                    "o1-mini (reasoning)",
    "o1-preview":                 "o1-preview (reasoning)",
    # Gemini 2.x
    "gemini-2.5-pro":             "Gemini 2.5 Pro",
    "gemini-2.5-flash":           "Gemini 2.5 Flash",
    "gemini-2.0-flash":           "Gemini 2.0 Flash",
    "gemini-1.5-pro":             "Gemini 1.5 Pro",
    "gemini-1.5-flash":           "Gemini 1.5 Flash",
}

# ─────────────────────────────────────────────
# API KEY PREFIX VALIDATION
# ─────────────────────────────────────────────
# Lightweight format checks — no network calls.

KEY_PREFIXES: Dict[str, str] = {
    "OpenAI":        "sk-",
    "Anthropic":     "sk-ant-",
    "Google Gemini": "AIza",   # Standard Google API key prefix
}


def validate_provider_config(provider: str, api_key: str) -> Tuple[bool, str]:
    """
    Validate an API key format without making any network calls.

    Returns:
        (is_valid: bool, message: str)

    Used by the Streamlit UI for instant sidebar feedback.  The backend
    may perform additional validation when the key is first used.
    """
    if provider == "Local (Ollama)":
        return True, "Ollama runs locally — no key needed."

    if not api_key or len(api_key.strip()) < 8:
        return False, "API key is missing or too short."

    key      = api_key.strip()
    expected = KEY_PREFIXES.get(provider, "")

    if expected and not key.startswith(expected):
        return False, (
            f"Key format mismatch for {provider} "
            f"(expected prefix: '{expected}')."
        )

    return True, f"{provider} key looks valid ✓"


def friendly_model_label(model_id: str) -> str:
    """Return a human-readable label for a model ID (falls back to the raw ID)."""
    return MODEL_LABELS.get(model_id, model_id)


# ─────────────────────────────────────────────
# GRAPH NODE LABELS  (used by the UI progress stepper)
# ─────────────────────────────────────────────

PIPELINE_NODES: List[str] = ["rewrite", "retrieve", "grade", "generate", "reflect"]

PIPELINE_NODE_LABELS: Dict[str, str] = {
    "rewrite":  "Rewrite",
    "retrieve": "Retrieve",
    "grade":    "Grade",
    "generate": "Generate",
    "reflect":  "Reflect",
}
