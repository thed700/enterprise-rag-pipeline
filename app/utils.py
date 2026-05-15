"""
utils.py — AuraRAG v3.2.0
Author: Akmal Raxmatov (github: thed700)

Changes v3.2.0:
  BUG-Q: setup_logging() always defaulted to INFO regardless of Settings.LOG_LEVEL.
          Now reads settings at call time so LOG_LEVEL=DEBUG actually works.
  NOTE:  lru_cache on get_settings() is intentional — Settings are immutable
         after process start (pydantic-settings reads .env once on construction).
         Cache avoids repeated disk reads. Call get_settings.cache_clear() in
         tests that need to vary env vars.
"""

import logging
import sys
from functools import lru_cache
from pydantic_settings import BaseSettings

APP_VERSION = "3.2.0"


class Settings(BaseSettings):
    # LLM provider keys (optional — users supply live in UI)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION: str = "aurarag"

    # Chunking (configurable, not hardcoded in engine)
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # Session memory
    SESSION_TTL_MINUTES: int = 60

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # BUG-Q fix: LOG_LEVEL is now *read* by setup_logging() instead of ignored
    LOG_LEVEL: str = "INFO"

    # CORS — comma-separated allowed origins
    ALLOWED_ORIGINS: str = "http://localhost:8501,http://127.0.0.1:8501"

    # File upload cap in MB (0 = no limit).
    MAX_UPLOAD_MB: int = 50

    # Rate limiting (requests/minute).
    RATE_LIMIT_QUERY: str = "30/minute"
    RATE_LIMIT_INGEST: str = "10/minute"

    # Streamlit
    STREAMLIT_PORT: int = 8501

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def setup_logging() -> None:
    """
    BUG-Q fix: reads LOG_LEVEL from Settings instead of always defaulting
    to INFO.  Call after get_settings() is ready (i.e. after .env is parsed).
    """
    settings = get_settings()
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,   # override any earlier basicConfig call (e.g. from uvicorn)
    )
    for noisy in ["httpx", "chromadb", "sentence_transformers", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
