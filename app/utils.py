"""
utils.py — AuraRAG v3.6
Author: Akmal Raxmatov (github: thed700)

Changes v3.6:
  - APP_VERSION bumped to "3.6".
  - No settings changes; all settings from v3.5 retained.
"""

import logging
import sys
from functools import lru_cache

from pydantic_settings import BaseSettings

APP_VERSION = "3.6"


class Settings(BaseSettings):
    # ── LLM provider keys (optional — users supply keys live in the UI) ─────
    OPENAI_API_KEY:    str = ""
    OPENAI_MODEL:      str = "gpt-4.1-mini"
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY:    str = ""

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION:  str = "aurarag"

    # ── Chunking ──────────────────────────────────────────────────────────────
    CHUNK_SIZE:    int = 512
    CHUNK_OVERLAP: int = 64

    # ── Session memory ────────────────────────────────────────────────────────
    SESSION_TTL_MINUTES: int = 60

    # ── API server ────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # BUG-Q fix (v3.2.0): LOG_LEVEL is read by setup_logging() instead of
    # being silently ignored (was always INFO regardless of env).
    LOG_LEVEL: str = "INFO"

    # ── CORS — comma-separated allowed origins ────────────────────────────────
    ALLOWED_ORIGINS: str = "http://localhost:8501,http://127.0.0.1:8501"

    # ── Upload cap in MB (0 = no limit) ───────────────────────────────────────
    MAX_UPLOAD_MB: int = 50

    # ── Rate limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_QUERY:  str = "30/minute"
    RATE_LIMIT_INGEST: str = "10/minute"

    # ── Streamlit ─────────────────────────────────────────────────────────────
    STREAMLIT_PORT: int = 8501

    # BUG-AE fix (v3.3): source snippet length was hardcoded to 300 chars;
    # now configurable via .env so operators can tune without code changes.
    SOURCE_SNIPPET_LEN: int = 300

    # ── v3.4 LangGraph pipeline settings ─────────────────────────────────────
    GRADE_THRESHOLD:    float = 0.5
    REFLECT_ENABLED:    bool  = True
    MAX_REFLECT_LOOPS:  int   = 1
    REWRITE_MAX_TOKENS: int   = 128
    GRADE_MAX_TOKENS:   int   = 64

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def setup_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    for noisy in ["httpx", "chromadb", "sentence_transformers", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
