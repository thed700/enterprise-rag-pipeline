# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — AuraRAG v3.6
# Author: Akmal Raxmatov (github: thed700)
#
# Changes v3.6:
#   - Version label bumped to 3.6.
#   - Removed unstructured from requirements (BUG-UNSTRUCTURED fix) which
#     eliminates ~2 GB of unnecessary deps (detectron2, tesseract, etc.)
#     and makes this image significantly faster to build on HF Spaces.
#   - Added writable tmp mount point for Chroma fallback path used on
#     read-only HF Spaces container filesystems.
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
        libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch first to avoid pulling the multi-GB GPU variant
RUN pip install --no-cache-dir \
        torch \
        --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    API_BASE=http://localhost:8000 \
    # Fallback Chroma dir for read-only container filesystems (HF Spaces)
    AURARAG_CACHE_DIR=/tmp/aurarag

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libmagic1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root system user for security compliance
RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source code with appropriate ownership
COPY --chown=appuser:appuser . .

# Initialize data directories and writable tmp cache
RUN mkdir -p /app/data/chroma_db /tmp/aurarag \
    && chown -R appuser:appuser /app/data /tmp/aurarag

RUN chmod +x /app/entrypoint.sh && chown appuser:appuser /app/entrypoint.sh

USER appuser

# Hugging Face Spaces strictly routes inbound traffic via port 7860
EXPOSE 7860

LABEL org.opencontainers.image.version="3.6.0" \
      org.opencontainers.image.title="AuraRAG" \
      org.opencontainers.image.authors="Akmal Raxmatov"

CMD ["./entrypoint.sh"]
