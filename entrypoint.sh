#!/bin/bash

# 1. Ensure the project root is in the Python path
export PYTHONPATH=$(pwd):$PYTHONPATH

# 2. Configure environment variables for runtime
export ALLOWED_ORIGINS="*"
export API_BASE="http://localhost:8000"

# 3. Start FastAPI backend on all interfaces inside the container
uvicorn main.py:app --host 0.0.0.0 --port 8000 --workers 1 &

# 4. Wait for the backend to be fully healthy before launching UI
echo "Waiting for FastAPI backend to start..."
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "Backend is up and running!"
        break
    fi
    sleep 1
done

# 5. Launch Streamlit with CORS protection flags disabled for Hugging Face proxy routing
streamlit run app/ui.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false