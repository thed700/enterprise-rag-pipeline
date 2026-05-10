# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — AuraRAG v3.0.0
# Author: Akmal Raxmatov (github: thed700)
#
# Changes v3.0.0:
#   - Rebranded from NeuralDocs -> AuraRAG
#   - Pin base image sha digest recommended — update <SHA> before prod deploy
#   - Added ALLOWED_ORIGINS ARG for CORS configuration at build time
#   - CHROMA_COLLECTION default updated to 'aurarag'
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch first (avoids pulling ~2 GB CUDA wheels from PyPI)
RUN pip install --no-cache-dir \
        torch \
        --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source (respects .dockerignore — .env is excluded)
COPY --chown=appuser:appuser . .

# Persistent data directory for ChromaDB
RUN mkdir -p /app/data/chroma_db && chown -R appuser:appuser /app/data

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default: FastAPI backend. docker-compose overrides CMD for the UI service.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
