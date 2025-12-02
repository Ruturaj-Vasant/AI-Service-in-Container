"""Training script for MNIST (CPU only).

Purpose
-------
- Train a small CNN on MNIST and save weights as `model.pth`.
- The output directory comes from the env var `MODEL_DIR` (defaults to `/mnt/model`).
  In Kubernetes, `/mnt/model` is a PersistentVolumeClaim (PVC) mount so the file
  survives after the training pod completes.

Key environment variables
-------------------------
- MODEL_DIR  : where to save the model (default `/mnt/model`)
- DATA_ROOT  : where to cache the MNIST dataset (default `/tmp/data`)
- EPOCHS     : number of training epochs (default 1)
- BATCH_SIZE : batch size (default 64)
- LR         : learning rate (default 0.001)
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(nn.Module):
    """Simple CNN sized for 1x28x28 MNIST images.

    Architecture: Conv(1->32,3x3) -> ReLU -> Conv(32->64,3x3) -> ReLU -> MaxPool(2)
    -> Dropout -> Flatten -> FC(9216->128) -> ReLU -> Dropout -> FC(128->10).
    """
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


def train(epochs: int = 1, batch_size: int = 64, lr: float = 1e-3):
    """Train for a few epochs and save weights.

    Uses CPU to be portable on GKE Autopilot nodes. Normalizes inputs with the
    standard MNIST mean/std so that training and inference preprocessing match.
    """
    device = torch.device("cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Where torchvision will store the raw MNIST files (inside the container)
    data_root = os.environ.get("DATA_ROOT", "/tmp/data")
    train_dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader, start=1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 0:
                avg = running_loss / 100
                print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] loss: {avg:.4f}", flush=True)
                running_loss = 0.0

    # Save into PVC mount (in-cluster) or local folder (when running locally)
    model_dir = Path(os.environ.get("MODEL_DIR", "/mnt/model"))
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pth"
    torch.save(model.state_dict(), str(model_path))
    print(f"Saved model to {model_path}", flush=True)


if __name__ == "__main__":
    # Read simple args via env vars for simplicity (keeps container CLI simple)
    epochs = int(os.environ.get("EPOCHS", "1"))
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))
    lr = float(os.environ.get("LR", "0.001"))
    train(epochs=epochs, batch_size=batch_size, lr=lr)
