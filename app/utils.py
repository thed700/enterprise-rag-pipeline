"""
utils.py — AuraRAG v3.1.0
Author: Akmal Raxmatov (github: thed700)

Changes v3.1.0:
  - APP_VERSION bumped to 3.1.0
  - Added CHUNK_SIZE / CHUNK_OVERLAP settings (configurable from .env)
  - Added SESSION_TTL_MINUTES setting
  - Added RATE_LIMIT_QUERY / RATE_LIMIT_INGEST for slowapi (BUG-H fix)
  - BUG-L fix: APP_VERSION is the single source of truth — models.py imports it
"""

import logging
import sys
from functools import lru_cache
from pydantic_settings import BaseSettings

APP_VERSION = "3.1.0"


def setup_logging(level: str = "INFO") -> None:
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    for noisy in ["httpx", "chromadb", "sentence_transformers", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Settings(BaseSettings):
    # LLM provider keys (optional — users supply live in UI)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION: str = "aurarag"

    # Chunking (BUG-C: configurable, not hardcoded in engine)
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # Session memory
    SESSION_TTL_MINUTES: int = 60

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # CORS — comma-separated allowed origins
    ALLOWED_ORIGINS: str = "http://localhost:8501,http://127.0.0.1:8501"

    # File upload cap in MB (0 = no limit). BUG-F fix applied in main.py.
    MAX_UPLOAD_MB: int = 50

    # Rate limiting (requests/minute). BUG-H fix.
    RATE_LIMIT_QUERY: str = "30/minute"
    RATE_LIMIT_INGEST: str = "10/minute"

    # Streamlit
    STREAMLIT_PORT: int = 8501

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
