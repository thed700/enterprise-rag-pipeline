"""
utils.py - Helper Functions: Logging, Config, Environment
Author: Akmal Raxmatov (github: thed700)

UPGRADES (v3.0.0):
  - APP_VERSION bumped to 3.0.0
  - Added ALLOWED_ORIGINS setting (comma-separated, used by CORSMiddleware)
  - Added MAX_UPLOAD_MB setting for ingest file-size cap
  - Rebranded from NeuralDocs -> AuraRAG in log output
"""

import logging
import sys
from functools import lru_cache
from pydantic_settings import BaseSettings

APP_VERSION = "3.0.0"


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for production."""
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    for noisy in ["httpx", "chromadb", "sentence_transformers"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # Provider keys — all optional; users supply them live via the UI.
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION: str = "aurarag"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # CORS — comma-separated list of allowed origins.
    # Default: Streamlit dev server. Override in production.
    ALLOWED_ORIGINS: str = "http://localhost:8501,http://127.0.0.1:8501"

    # Max file size accepted by /ingest (MB). 0 = no limit.
    MAX_UPLOAD_MB: int = 50

    # Streamlit
    STREAMLIT_PORT: int = 8501

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
