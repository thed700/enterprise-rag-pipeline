# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — AuraRAG v3.4.0
# Author: Akmal Raxmatov (github: thed700)
#
# Changes v3.4.0:
#   - Migrated setup for Hugging Face Spaces deployment.
#   - Exposed port 7860 to match Hugging Face default container routing.
#   - Introduced entrypoint.sh orchestration to run both FastAPI and Streamlit.
#   - Retained libmagic1 runtime dependencies (BUG-W fix) and multi-stage layout.
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
    API_BASE=http://localhost:8000

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root system user for security compliance
RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source code with appropriate ownership
COPY --chown=appuser:appuser . .

# Initialize data directories and explicitly set user permissions
RUN mkdir -p /app/data/chroma_db && chown -R appuser:appuser /app/data

# ATTENTION: Grant execution permissions for the entrypoint orchestration script
RUN chmod +x /app/entrypoint.sh && chown appuser:appuser /app/entrypoint.sh

USER appuser

# Hugging Face Spaces strictly routes inbound traffic via port 7860
EXPOSE 7860

LABEL org.opencontainers.image.version="3.4.0" \
      org.opencontainers.image.title="AuraRAG" \
      org.opencontainers.image.authors="Akmal Raxmatov"

# Execute the custom entrypoint script instead of a single service worker
CMD ["./entrypoint.sh"]