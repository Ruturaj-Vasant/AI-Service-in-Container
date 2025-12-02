"""FastAPI inference server for MNIST (CPU only).

Purpose
-------
- Load the trained weights from `MODEL_DIR/model.pth` on startup.
- Provide a minimal web UI at `/` and a POST `/predict` endpoint that accepts
  an image file and returns the predicted class + softmax probabilities.

Key environment variables
-------------------------
- MODEL_DIR : directory containing `model.pth` (default `/mnt/model`). In-cluster
  this is the same PVC mount path used by training.
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class Net(nn.Module):
    """Same architecture used during training (must match shapes)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def load_model(model_path: Path) -> Net:
    """Load weights from disk into a fresh Net and set eval mode."""
    model = Net()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    state = torch.load(str(model_path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/mnt/model"))
MODEL_PATH = MODEL_DIR / "model.pth"

app = FastAPI(title="MNIST Inference Service", version="1.0")
model: Optional[Net] = None


@app.on_event("startup")
def _load_on_start():
    """Load the model once when the process starts."""
    global model
    model = load_model(MODEL_PATH)


@app.get("/healthz")
def healthz():
    """Lightweight readiness/liveness check for probes."""
    ok = MODEL_PATH.exists()
    return {"status": "ok" if ok else "no-model"}


@app.get("/", response_class=HTMLResponse)
def index():
    """Tiny HTML form to upload an image to /predict from a browser."""
    return """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>MNIST Inference</title>
        <style>
          body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }
          .card { max-width: 520px; padding: 1rem 1.25rem; border: 1px solid #e5e7eb; border-radius: 8px; }
          button { padding: .5rem .75rem; border: 1px solid #e5e7eb; border-radius: 6px; background: #111827; color: white; cursor: pointer; }
          pre { background: #0b1021; color: #e5e7eb; padding: .75rem; border-radius: 6px; overflow:auto; }
          .row { display:flex; gap:.75rem; align-items:center; }
        </style>
      </head>
      <body>
        <h2>MNIST Inference</h2>
        <div class="card">
          <form id="f" class="row">
            <input type="file" name="file" accept="image/*" required />
            <button type="submit">Predict</button>
          </form>
          <div id="status" style="margin-top:.75rem;color:#374151"></div>
          <pre id="out" style="margin-top:.75rem"></pre>
        </div>
        <script>
          const f = document.getElementById('f');
          const out = document.getElementById('out');
          const status = document.getElementById('status');
          f.addEventListener('submit', async (e) => {
            e.preventDefault();
            out.textContent = '';
            status.textContent = 'Uploading...';
            const form = new FormData(f);
            try {
              const res = await fetch('/predict', { method: 'POST', body: form });
              const txt = await res.text();
              status.textContent = res.ok ? 'OK' : `HTTP ${res.status}`;
              try { out.textContent = JSON.stringify(JSON.parse(txt), null, 2); }
              catch { out.textContent = txt; }
            } catch (err) {
              status.textContent = 'Request failed';
              out.textContent = String(err);
            }
          });
        </script>
      </body>
    </html>
    """


transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Run inference on an uploaded image.

    Accepts multipart form field `file`. Converts to 1x28x28 grayscale,
    applies the same normalization as training, and returns JSON.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("L")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Prepare tensor of shape [B,C,H,W] = [1,1,28,28]
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())
        probs = torch.softmax(logits, dim=1).squeeze().tolist()

    return JSONResponse({"prediction": pred, "probs": probs})
