# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — AuraRAG v3.3
# Author: Akmal Raxmatov (github: thed700)
#
# Changes v3.3:
#   - Version label bumped to 3.3
#   - Builder stage: added 'file' and 'libmagic1' for autodetect_encoding in
#     TextLoader (BUG-W fix — required by python-magic on some distros).
#   - Runner stage: libmagic1 carried through for runtime use.
#
# Retained from v3.1.0:
#   BUG-N: HEALTHCHECK defined per-service in docker-compose.yml, not here.
#   CPU-only torch installed from PyTorch wheel index (avoids 2 GB CUDA build).
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# BUG-W fix: libmagic1 needed by python-magic (autodetect_encoding in TextLoader)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libmagic1 \
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

# BUG-W fix: libmagic1 required at runtime for autodetect_encoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libmagic1 \
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

LABEL org.opencontainers.image.version="3.3" \
      org.opencontainers.image.title="AuraRAG" \
      org.opencontainers.image.authors="Akmal Raxmatov"

# HEALTHCHECK defined per-service in docker-compose.yml (BUG-N)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
