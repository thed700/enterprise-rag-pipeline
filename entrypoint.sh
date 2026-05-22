#!/bin/bash
set -e

# 1. Ensure the project root is in the Python path
export PYTHONPATH=$(pwd):$PYTHONPATH

# 2. Configure environment variables for runtime
export ALLOWED_ORIGINS="*"
export API_BASE="http://localhost:8000"

# 3. Start FastAPI backend — module path must be app.main:app
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 &

# 4. Wait for the backend to be fully healthy before launching UI
echo "Waiting for FastAPI backend to start..."
for i in {1..30}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "Backend is up and running!"
        break
    fi
    echo "  attempt $i/30..."
    sleep 2
done

# 5. Launch Streamlit on HF Spaces port 7860
streamlit run app/ui.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
