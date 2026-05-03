# ─────────────────────────────────────────────
# Dockerfile — Enterprise RAG System
# Author: Akmal Raxmatov (github: thed700)
# ─────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /rag-system

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY app/ ./app/
COPY tests/ ./tests/
COPY .env.example .env.example

# Create data directory
RUN mkdir -p data/chroma_db data/sample_docs

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Default: start FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
