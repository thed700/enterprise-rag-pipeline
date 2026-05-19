#!/bin/bash

# 1. Setup explicit environment paths
export PYTHONPATH="/app:/app/app:$PYTHONPATH"
export ALLOWED_ORIGINS="*"
export API_BASE="http://localhost:8000"

# 2. Start FastAPI on port 8000 binding to all interfaces
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 &

# 3. Wait for immediate 200 OK response from our lightweight health check
echo "Waiting for backend routing loop..."
for i in {1..15}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "FastAPI route mapped successfully!"
        break
    fi
    echo "Ping retry $i..."
    sleep 1
done

# 4. Launch Streamlit interface with proxy security disabled
streamlit run app/ui.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false