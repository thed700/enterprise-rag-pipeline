"""
utils.py — AuraRAG v3.5
Author: Akmal Raxmatov (github: thed700)

Changes v3.5:
  - APP_VERSION bumped to "3.5".
  - Bug fixes BUG-AK through BUG-AO (see CHANGELOG.md).
  - Default OPENAI_MODEL updated to gpt-4.1-mini.

Retained from v3.4:
  GRADE_THRESHOLD, REFLECT_ENABLED, MAX_REFLECT_LOOPS,
  REWRITE_MAX_TOKENS, GRADE_MAX_TOKENS.
Retained from v3.3 / v3.2.0:
  BUG-Q:  setup_logging() reads LOG_LEVEL from Settings (not always INFO).
  BUG-AC: lru_cache on get_settings() — Settings are immutable after process
          start. Call get_settings.cache_clear() in tests that vary env vars.
  BUG-AE: SOURCE_SNIPPET_LEN configurable via .env (was hardcoded to 300).
"""

import logging
import sys
from functools import lru_cache

from pydantic_settings import BaseSettings

APP_VERSION = "3.5"


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

    # Relevance score threshold for the Document Grader node.
    # Chunks with a grader score below this value are filtered out.
    # Range: 0.0 – 1.0. Default 0.5 (medium strictness).
    GRADE_THRESHOLD: float = 0.5

    # Enable/disable the Reflect / Self-Correction node.
    # When False the graph exits directly after Generate.
    REFLECT_ENABLED: bool = True

    # Maximum Reflect→Retrieve loops before forcing a final answer.
    # Prevents infinite correction on pathological inputs.
    MAX_REFLECT_LOOPS: int = 1

    # Token budget for the lightweight Query Rewrite LLM call.
    # Output is a short query string — keep small.
    REWRITE_MAX_TOKENS: int = 128

    # Token budget for each Document Grader LLM call.
    # Calls are parallelised per chunk — keep small to control latency cost.
    GRADE_MAX_TOKENS: int = 64

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    lru_cache is intentional — pydantic-settings reads .env once at
    construction time; repeated calls return the same instance with zero
    disk I/O.  Tests that need to vary env vars must call
    get_settings.cache_clear() before constructing a new Settings.
    """
    return Settings()


def setup_logging() -> None:
    """
    Configure the root logger.

    BUG-Q fix (v3.2.0): reads LOG_LEVEL from Settings instead of always
    defaulting to INFO.  Call once at module load time in main.py, after
    get_settings() has parsed the .env file.
    """
    settings = get_settings()
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # override any earlier basicConfig call (e.g. from uvicorn)
    )
    # Suppress noisy third-party loggers at WARNING level regardless of
    # the configured LOG_LEVEL.
    for noisy in ["httpx", "chromadb", "sentence_transformers", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
