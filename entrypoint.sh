#!/bin/bash
set -e

# ─────────────────────────────────────────────────────────────────────────────
# entrypoint.sh — AuraRAG v3.6
# Orchestrates FastAPI backend + Streamlit frontend for HF Spaces / Docker.
# ─────────────────────────────────────────────────────────────────────────────

# 1. Ensure the project root is in the Python path
export PYTHONPATH=$(pwd):$PYTHONPATH

# 2. Configure environment variables for runtime
export ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-*}"
export API_BASE="${API_BASE:-http://localhost:8000}"

# 3. Ensure writable cache directory exists (needed on HF Spaces read-only FS)
export AURARAG_CACHE_DIR="${AURARAG_CACHE_DIR:-/tmp/aurarag}"
mkdir -p "$AURARAG_CACHE_DIR"

# 4. Start FastAPI backend in the background
echo "[entrypoint] Starting FastAPI backend on port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 &
BACKEND_PID=$!

# 5. Wait for the backend to be healthy before launching the UI
echo "[entrypoint] Waiting for backend to be ready..."
MAX_WAIT=60
WAITED=0
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "[entrypoint] ERROR: backend did not start within ${MAX_WAIT}s"
        kill "$BACKEND_PID" 2>/dev/null || true
        exit 1
    fi
    echo "[entrypoint]   ... waiting (${WAITED}s)"
    sleep 2
    WAITED=$((WAITED + 2))
done
echo "[entrypoint] Backend is healthy."

# 6. Launch Streamlit on HF Spaces port 7860
echo "[entrypoint] Starting Streamlit UI on port 7860..."
exec streamlit run app/ui.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
