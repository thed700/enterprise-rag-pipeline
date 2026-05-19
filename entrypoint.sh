#!/bin/bash

# 1. Append the current working directory to PYTHONPATH.
# This prevents potential "ModuleNotFoundError" issues inside the container.
export PYTHONPATH=$(pwd):$PYTHONPATH

# 2. Start the FastAPI backend engine in the background.
# Port 8000 is retained so the Streamlit UI can internally communicate with the API.
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 &

# 3. Allow a brief buffer (3 seconds) for the backend service to initialize completely.
sleep 3

# 4. Launch the Streamlit frontend application as the foreground process.
# Port 7860 is explicitly set to comply with Hugging Face Spaces inbound routing requirements.
streamlit run app/ui.py --server.port 7860 --server.address 0.0.0.0 --server.headless true