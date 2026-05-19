#!/bin/bash

# 1. Force the current container workdir into Python path
export PYTHONPATH="/app:$PYTHONPATH"

# 2. Configure critical environment variables for Hugging Face proxy routing
export ALLOWED_ORIGINS="*"
export API_BASE="http://localhost:8000"

# 3. Start FastAPI backend explicitly using python -m uvicorn to avoid path mismatch
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 &

# 4. Wait for the backend to respond with a 200 OK health status
echo "Waiting for FastAPI backend to spin up..."
for i in {1..15}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "Backend connected successfully on port 8000!"
        break
    fi
    echo "Attempt $i: Backend not ready yet. Retrying..."
    sleep 1
done

# 5. Launch Streamlit frontend with explicit proxy safety flags for Hugging Face Spaces
streamlit run app/ui.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false