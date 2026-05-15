"""
constants.py — AuraRAG v3.2.0
Shared constants used by both the engine (backend) and the UI (frontend).
Extracted from engine.py so the Streamlit container does NOT need to import
heavy ML dependencies (torch, sentence-transformers, chromadb) just to render
the provider selector.

Author: Akmal Raxmatov (github: thed700)

Changes v3.2.0:
  BUG-R: Anthropic model list updated to current stable IDs (claude-opus-4-6,
          claude-sonnet-4-6 — these are the correct v4.6 family IDs).
          Removed stale/non-existent claude-opus-4-5 / claude-sonnet-4-5 aliases.
  BUG-R: claude-haiku-4-5-20251001 retained (date-stamp required by API).
  BUG-R: Added latest OpenAI o-series and Gemini 2.5 models.
"""

from typing import Dict, List, Tuple

# ─────────────────────────────────────────────
# PROVIDER → MODEL REGISTRY
# ─────────────────────────────────────────────

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
        "claude-opus-4-6",           # BUG-R: correct v4.6 family ID
        "claude-sonnet-4-6",         # BUG-R: correct v4.6 family ID
        "claude-haiku-4-5-20251001", # full date-stamped ID required by the API
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
    ],
    "Google Gemini": [
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
    "Local (Ollama)": [
        "llama3",
        "llama3:8b",
        "llama3:70b",
        "mistral",
        "mixtral",
        "phi3",
        "gemma2",
    ],
}

# Human-readable labels for models with long date-stamped IDs
MODEL_LABELS: Dict[str, str] = {
    "claude-haiku-4-5-20251001":  "claude-haiku-4-5",
    "claude-3-5-sonnet-20241022": "claude-3.5-sonnet",
    "claude-3-opus-20240229":     "claude-3-opus",
}

# API key prefix validation map
KEY_PREFIXES: Dict[str, str] = {
    "OpenAI":        "sk-",
    "Anthropic":     "sk-ant-",
    "Google Gemini": "AI",
}


def validate_provider_config(provider: str, api_key: str) -> Tuple[bool, str]:
    """Lightweight key format validator — no network calls, no ML imports."""
    if provider == "Local (Ollama)":
        return True, "Ollama runs locally — no key needed."
    if not api_key or len(api_key.strip()) < 8:
        return False, "API key appears to be missing or too short."
    key = api_key.strip()
    expected = KEY_PREFIXES.get(provider, "")
    if expected and not key.startswith(expected):
        return False, (
            f"Key doesn't match expected {provider} format "
            f"(should start with '{expected}')."
        )
    return True, f"{provider} key looks valid ✓"


def friendly_model_label(model_id: str) -> str:
    """Return a human-readable label for a model ID."""
    return MODEL_LABELS.get(model_id, model_id)
