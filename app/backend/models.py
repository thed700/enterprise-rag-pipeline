"""
models.py — shared provider/model registry for AuraRAG.

This module is intentionally lightweight so both FastAPI and Streamlit can
import it without pulling in any ML dependencies.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

PROVIDER_MODELS: Dict[str, List[str]] = {
    "OpenAI": [
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "Anthropic": [
        "claude-opus-4-5",
        "claude-sonnet-4-5",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ],
    "Google Gemini": [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ],
    # Keep the existing provider label for compatibility with the rest of
    # the codebase and the UI, but treat the model as free-form input.
    "Local (Ollama)": [
        "llama3",
        "mistral",
    ],
}

# Human-readable aliases only where the raw model IDs are noisy.
MODEL_LABELS: Dict[str, str] = {
    "claude-opus-4-5": "Claude Opus 4.5",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
    "claude-3-5-haiku-20241022": "Claude 3.5 Haiku",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gpt-4o-mini": "GPT-4o mini",
}

KEY_PREFIXES: Dict[str, str] = {
    "OpenAI": "sk-",
    "Anthropic": "sk-ant-",
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
