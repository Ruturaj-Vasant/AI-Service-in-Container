# AI-Service-in-Container

# MNIST on GKE (Training + Inference)

This project trains an MNIST model in Kubernetes (Job) and serves predictions via FastAPI (Deployment + Service) on GKE Autopilot. A PersistentVolumeClaim stores the trained model between phases.

## Images

- Training image: `asia-east1-docker.pkg.dev/csci-ga-3033085/mnist-repo/mnist-train:latest`
- Inference image: `asia-east1-docker.pkg.dev/csci-ga-3033085/mnist-repo/mnist-infer:latest`

## Build & Push

```bash
export PROJECT_ID=csci-ga-3033085
export REGION=asia-east1
export REPO=mnist-repo
export TRAIN_IMG=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/mnist-train:latest
export INFER_IMG=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/mnist-infer:latest

# If on Apple Silicon
export DOCKER_DEFAULT_PLATFORM=linux/amd64

# Build
docker build -t $TRAIN_IMG -f training/Dockerfile .
docker build -t $INFER_IMG -f inference/Dockerfile .

# Push
docker push $TRAIN_IMG
docker push $INFER_IMG
```

## Kubernetes Apply

```bash
# PVC
kubectl apply -f k8s/pvc.yaml

# Run training Job (writes /mnt/model/model.pth)
kubectl apply -f k8s/job-training.yaml
kubectl logs -l job-name=mnist-training-job -f

# After Job completes, deploy inference
kubectl apply -f k8s/deploy-inference.yaml
kubectl apply -f k8s/svc-inference.yaml

# Get external IP
kubectl get svc mnist-inference-svc -w
```

## Test Prediction

```bash
EXTERNAL_IP=$(kubectl get svc mnist-inference-svc -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl -s -X POST \
  -F "file=@/path/to/some_mnist_like_digit.png" \
  http://$EXTERNAL_IP/predict | jq
```

## Files

- `training/training.py` — trains CNN and saves `/mnt/model/model.pth`
- `training/Dockerfile` — CPU PyTorch training image
- `inference/inference.py` — FastAPI server loading saved model
- `inference/Dockerfile` — CPU PyTorch + FastAPI image
- `k8s/pvc.yaml` — PersistentVolumeClaim for model storage
- `k8s/job-training.yaml` — Kubernetes Job for one-time training
- `k8s/deploy-inference.yaml` — Deployment for inference server
- `k8s/svc-inference.yaml` — LoadBalancer Service exposing `/predict`
