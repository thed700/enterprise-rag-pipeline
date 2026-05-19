#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH

# Hugging Face-da CORS 403 xatoligini oldini olish uchun hamma origin-larga ruxsat beramiz
export ALLOWED_ORIGINS="*"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 &
sleep 3
streamlit run app/ui.py --server.port 7860 --server.address 0.0.0.0 --server.headless true