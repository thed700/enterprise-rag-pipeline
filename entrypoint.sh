#!/bin/bash

# 1. Ensure the project root is in the Python path
export PYTHONPATH=$(pwd):$PYTHONPATH

# 2. Fix CORS and Route constraints globally for Hugging Face
# This allows the Streamlit frontend to fully talk to FastAPI without 403 blocks
export ALLOWED_ORIGINS="*"
export API_BASE="http://127.0.0.1:8000"

# 3. Start FastAPI backend engine on local loopback interface
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 &

# 4. Wait for the backend to be fully healthy before launching UI
echo "Waiting for FastAPI backend to start..."
for i in {1..10}; do
    if curl -s http://127.0.0.1:8000/health > /dev/null; then
        echo "Backend is up and running!"
        break
    fi
    sleep 1
done

# 5. Launch Streamlit frontend targeting the exact local loopback API
streamlit run app/ui.py --server.port 7860 --server.address 0.0.0.0 --server.headless true