#!/bin/bash
set -e

# 1. Ensure the project root is in the Python path
export PYTHONPATH=$(pwd):$PYTHONPATH

# 2. Configure environment variables for runtime
export ALLOWED_ORIGINS="*"
export API_BASE="http://127.0.0.1:8000" # localhost o'rniga 127.0.0.1 ishlatildi

# 3. Start FastAPI backend and save its Process ID
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 &
BACKEND_PID=$!

# 4. Wait for the backend to be fully healthy before launching UI
echo "Waiting for FastAPI backend to start..."
BACKEND_UP=false

for i in {1..30}; do
    # Server haliyam ishlayotganini tekshirish
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "❌ FastAPI backend ishga tushmadi yoki qulab tushdi! Jurnallarni (Logs) tekshiring."
        exit 1
    fi

    # Sog'lomlik holatini tekshirish
    if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "✅ Backend muvaffaqiyatli ishga tushdi!"
        BACKEND_UP=true
        break
    fi
    echo "  urinish $i/30..."
    sleep 2
done

if [ "$BACKEND_UP" = false ]; then
    echo "❌ FastAPI backend o'z vaqtida javob bermadi."
    exit 1
fi

# 5. Launch Streamlit on HF Spaces port 7860
streamlit run app/ui.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false