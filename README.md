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

## Steps Involved (End-to-End)

The quickest, reliable path from zero to a public endpoint. Replace project/region as needed.

1) Prerequisites

```bash
gcloud --version
kubectl version --client
docker --version

# Required for kubectl → GKE auth
export USE_GKE_GCLOUD_AUTH_PLUGIN=True
gcloud components install gke-gcloud-auth-plugin || brew install gke-gcloud-auth-plugin
```

2) Project setup

```bash
export PROJECT_ID=csci-ga-3033085
export REGION=asia-east1
export REPO=mnist-repo

gcloud auth login
gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION
gcloud services enable container.googleapis.com artifactregistry.googleapis.com
```

3) Artifact Registry

```bash
gcloud artifacts repositories create $REPO \
  --repository-format=docker --location=$REGION || true
gcloud auth configure-docker $REGION-docker.pkg.dev

export TRAIN_IMG=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/mnist-train:latest
export INFER_IMG=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/mnist-infer:latest
```

4) GKE Autopilot cluster

```bash
gcloud container clusters create-auto mnist-cluster --region $REGION
gcloud container clusters get-credentials mnist-cluster --region $REGION --project $PROJECT_ID
kubectl get storageclass
```

5) Build and push images

```bash
# Apple Silicon (Mac) compatibility
export DOCKER_DEFAULT_PLATFORM=linux/amd64

docker build -t $TRAIN_IMG -f training/Dockerfile . && docker push $TRAIN_IMG
docker build -t $INFER_IMG -f inference/Dockerfile . && docker push $INFER_IMG
```

6) Persist model with a Job

```bash
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/job-training.yaml
kubectl logs -l job-name=mnist-training-job -f  # wait for "Saved model to /mnt/model/model.pth"
```

7) Serve inference

```bash
kubectl apply -f k8s/deploy-inference.yaml

# Option A: Public IP (costs while active)
kubectl apply -f k8s/svc-inference.yaml
kubectl get svc mnist-inference-svc -w   # note EXTERNAL-IP

# Option B: Free, on-demand access
# kubectl port-forward deploy/mnist-inference-deploy 8080:8080
# Then use: http://127.0.0.1:8080
```

8) Test

```bash
EXTERNAL_IP=$(kubectl get svc mnist-inference-svc -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl -s http://$EXTERNAL_IP/healthz
python3 samples/gen_samples.py
curl -s -X POST -F "file=@samples/zero.png" http://$EXTERNAL_IP/predict | jq
```

9) Cost control

```bash
# Remove LB but keep app
kubectl delete svc mnist-inference-svc

# Scale app to zero
kubectl scale deploy mnist-inference-deploy --replicas=0
```

10) Clean up to $0

```bash
kubectl delete svc mnist-inference-svc --ignore-not-found
kubectl delete deploy mnist-inference-deploy --ignore-not-found
kubectl delete job mnist-training-job --ignore-not-found
kubectl delete pvc mnist-model-pvc --ignore-not-found

gcloud container clusters delete mnist-cluster --region $REGION --quiet
gcloud artifacts repositories delete $REPO --location=$REGION --quiet
# Optional: gsutil rm -r gs://$PROJECT_ID\_cloudbuild
```

Notes:
- PVC is the durable disk; training writes `/mnt/model/model.pth`, inference reads it.
- LoadBalancer costs while active; port-forward avoids that.
- Apple Silicon must build `linux/amd64` images to run on GKE nodes.

## Practical Starter Path (with the “Why”)

Below is a practical starter path with the “why” behind every step.

### Start Here
- Goal: train once, save a model file, then serve predictions.
- Trick: pods are ephemeral; a PVC gives you durable storage at the same mount path (`/mnt/model`) for both training and inference.

### Local Dev: Training (prove code works before K8s)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow

export MODEL_DIR=$(pwd)/local_model_dir
python training/training.py  # should print: Saved model to local_model_dir/model.pth
```

### Local Dev: Inference (load the local model and predict)
```bash
pip install fastapi uvicorn[standard] python-multipart
export MODEL_DIR=$(pwd)/local_model_dir
uvicorn inference.inference:app --host 0.0.0.0 --port 8080
# In another terminal
python3 samples/gen_samples.py
curl -s -X POST -F "file=@samples/zero.png" http://127.0.0.1:8080/predict
```
Why: validates preprocessing, model load, and the /predict contract without containers.

### Containerize (Docker) so K8s can run it
```bash
export DOCKER_DEFAULT_PLATFORM=linux/amd64   # Apple Silicon tip
docker build -t mnist-train:local -f training/Dockerfile .
docker build -t mnist-infer:local -f inference/Dockerfile .

# Optional container tests using a host-mounted model dir
mkdir -p $(pwd)/local_model_dir
docker run --rm -e MODEL_DIR=/mnt/model -v $(pwd)/local_model_dir:/mnt/model mnist-train:local
docker run --rm -p 8080:8080 -e MODEL_DIR=/mnt/model -v $(pwd)/local_model_dir:/mnt/model mnist-infer:local
```

### Image Registry (what and how)
- What: a place GKE nodes pull images from (Artifact Registry on GCP).
```bash
export PROJECT_ID=csci-ga-3033085
export REGION=asia-east1
export REPO=mnist-repo
gcloud auth login && gcloud config set project $PROJECT_ID
gcloud services enable artifactregistry.googleapis.com
gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION || true
gcloud auth configure-docker $REGION-docker.pkg.dev

export TRAIN_IMG=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/mnist-train:latest
export INFER_IMG=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/mnist-infer:latest
docker tag mnist-train:local $TRAIN_IMG && docker push $TRAIN_IMG
docker tag mnist-infer:local $INFER_IMG && docker push $INFER_IMG
```

### Kubernetes (what “configure k8s” means)
- Create a cluster and let kubectl talk to it via kubeconfig.
```bash
gcloud services enable container.googleapis.com
gcloud container clusters create-auto mnist-cluster --region $REGION
gcloud container clusters get-credentials mnist-cluster --region $REGION
export USE_GKE_GCLOUD_AUTH_PLUGIN=True
```
- Apply manifests in this order: PVC → Job → Deployment → Service.
```bash
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/job-training.yaml
kubectl logs -l job-name=mnist-training-job -f  # wait for Saved model

kubectl apply -f k8s/deploy-inference.yaml
# Public LB (costs) or port-forward (free):
kubectl apply -f k8s/svc-inference.yaml || true
```

### Test in cluster
```bash
EXTERNAL_IP=$(kubectl get svc mnist-inference-svc -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl -s http://$EXTERNAL_IP/healthz
curl -s -X POST -F "file=@samples/zero.png" http://$EXTERNAL_IP/predict | jq
```

### Clean up to $0 when done
```bash
kubectl delete svc mnist-inference-svc --ignore-not-found
kubectl delete deploy mnist-inference-deploy --ignore-not-found
kubectl delete job mnist-training-job --ignore-not-found
kubectl delete pvc mnist-model-pvc --ignore-not-found
gcloud container clusters delete mnist-cluster --region $REGION --quiet
gcloud artifacts repositories delete $REPO --location=$REGION --quiet
```
