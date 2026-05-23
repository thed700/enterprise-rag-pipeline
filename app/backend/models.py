"""
models.py — shared provider/model registry for AuraRAG v3.6.

Changes v3.6:
  BUG-R fix: Corrected Anthropic model strings. Previous versioned IDs
  (claude-opus-4-5-20251101, claude-sonnet-4-5-20251022, etc.) are not valid
  API identifiers. Replaced with the correct short-form model strings that the
  Anthropic API actually accepts.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

PROVIDER_MODELS: Dict[str, List[str]] = {
    "OpenAI": [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "Anthropic": [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ],
    "Google Gemini": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
    "Local (Ollama)": [
        "llama3",
        "mistral",
    ],
}

MODEL_LABELS: Dict[str, str] = {
    "gpt-4.1":                    "GPT-4.1",
    "gpt-4.1-mini":               "GPT-4.1 mini",
    "gpt-4o-mini":                "GPT-4o mini",
    "claude-opus-4-6":            "Claude Opus 4.6",
    "claude-sonnet-4-6":          "Claude Sonnet 4.6",
    "claude-sonnet-4-5":          "Claude Sonnet 4.5",
    "claude-haiku-4-5":           "Claude Haiku 4.5",
    "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
    "claude-3-5-haiku-20241022":  "Claude 3.5 Haiku",
    "gemini-2.5-pro":             "Gemini 2.5 Pro",
    "gemini-2.5-flash":           "Gemini 2.5 Flash",
    "gemini-2.0-flash":           "Gemini 2.0 Flash",
}

KEY_PREFIXES: Dict[str, str] = {
    "OpenAI":        "sk-",
    "Anthropic":     "sk-ant-",
    "Google Gemini": "AIza",
}

OLLAMA_PROVIDER_LABEL = "Local (Ollama)"


def is_ollama_provider(provider: str) -> bool:
    return provider == OLLAMA_PROVIDER_LABEL


def provider_model_options(provider: str) -> List[str]:
    return list(PROVIDER_MODELS.get(provider, []))


def validate_provider_config(provider: str, api_key: str) -> Tuple[bool, str]:
    """
    Lightweight key-format validator. No network calls, no heavy imports.
    """
    if is_ollama_provider(provider):
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
    return MODEL_LABELS.get(model_id, model_id)
