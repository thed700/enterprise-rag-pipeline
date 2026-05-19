#!/bin/bash

# 1. Set direct root application paths
export PYTHONPATH=".:$PYTHONPATH"
export ALLOWED_ORIGINS="*"
export API_BASE="http://127.0.0.1:8000"

# 2. Start FastAPI backend directly using module context execution
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 &

# 3. Wait for the backend to respond with a 200 OK health status
echo "Waiting for FastAPI backend to spin up..."
for i in {1..20}; do
    if curl -s http://127.0.0.1:8000/health > /dev/null; then
        echo "Backend connected successfully on port 8000!"
        break
    fi
    echo "Attempt $i: Backend not ready yet. Retrying..."
    sleep 1
done

# 4. Launch Streamlit frontend with explicit proxy safety flags for Hugging Face Spaces
streamlit run app/ui.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false