"""
app.backend — lightweight backend namespace for AuraRAG.

Keep this module minimal so importing `app.backend.models` does not pull in
any document loaders or other optional dependencies.
"""

from .models import (
    PROVIDER_MODELS,
    MODEL_LABELS,
    KEY_PREFIXES,
    validate_provider_config,
    friendly_model_label,
    is_ollama_provider,
    provider_model_options,
)

__all__ = [
    "PROVIDER_MODELS",
    "MODEL_LABELS",
    "KEY_PREFIXES",
    "validate_provider_config",
    "friendly_model_label",
    "is_ollama_provider",
    "provider_model_options",
]
