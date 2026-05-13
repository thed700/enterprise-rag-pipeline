# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — AuraRAG v3.1.0
# Author: Akmal Raxmatov (github: thed700)
#
# Changes v3.1.0:
#   BUG-N: HEALTHCHECK removed from here — moved to docker-compose per-service
#          so the API and UI containers can use different ports.
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

# CPU-only torch — avoids the 2 GB CUDA wheels from the default PyPI index
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

# Non-root user
RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Exclude .env, data/, __pycache__ via .dockerignore
COPY --chown=appuser:appuser . .

RUN mkdir -p /app/data/chroma_db && chown -R appuser:appuser /app/data

USER appuser

EXPOSE 8000 8501

# BUG-N fix: HEALTHCHECK removed — defined per-service in docker-compose.yml
# Default CMD: FastAPI backend. docker-compose overrides for the UI service.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
