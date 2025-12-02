# CPU-only PyTorch training image
# - Installs Python + CPU builds of torch/vision/audio
# - Copies training script
# - Defaults to saving model into /mnt/model (PVC mount in Kubernetes)
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

# Install CPU-only PyTorch + torchvision
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
# Only the script is needed at runtime
COPY training/training.py /app/training.py

# Defaults; override via env in the Job if desired
ENV EPOCHS=1 BATCH_SIZE=64 LR=0.001

CMD ["python", "-u", "training.py"]
