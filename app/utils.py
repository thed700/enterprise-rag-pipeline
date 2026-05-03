"""
utils.py - Helper Functions: Logging, Config, Environment
Author: Akmal Raxmatov (github: thed700)
"""

import logging
import sys
from functools import lru_cache
from pydantic_settings import BaseSettings


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
    # Silence noisy third-party loggers
    for noisy in ["httpx", "chromadb", "sentence_transformers"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # OpenAI
    OPENAI_API_KEY: str = "sk-replace-me"
    OPENAI_MODEL: str = "gpt-4o-mini"

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION: str = "enterprise_rag"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Streamlit
    STREAMLIT_PORT: int = 8501

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
