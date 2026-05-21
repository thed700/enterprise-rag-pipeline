"""
constants.py — AuraRAG compatibility shim.

Legacy imports still resolve through this module. The canonical provider
registry now lives in app.backend.models.
"""

from app.backend.models import (
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
