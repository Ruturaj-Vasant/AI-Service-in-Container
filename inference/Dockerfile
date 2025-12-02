# CPU-only PyTorch + FastAPI inference image
# - Installs Python + CPU torch/vision/audio
# - Installs FastAPI + Uvicorn and minimal deps for image upload
# - Runs the server on port 8080
FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/mnt/model

# Minimal build tools; keep image small
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch and FastAPI stack
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] pillow python-multipart

WORKDIR /app
# Only the server file; no extra sources needed
COPY inference/inference.py /app/inference.py

EXPOSE 8080
# Start FastAPI via Uvicorn
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8080"]
