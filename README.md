# AI-Service-in-Container

End-to-end MNIST on GKE: a one-shot Job trains a CNN and saves `model.pth` to a PersistentVolumeClaim; a FastAPI server loads the model and exposes `/predict` (with a tiny upload UI).

## Screenshots

| Upload UI | Prediction Result |
| --- | --- |
| <img src="Report/MNIST_upload.png" width="420" /> | <img src="Report/MNIST_result.png" width="420" /> |

## Files

- `training/training.py` — trains CNN and saves `/mnt/model/model.pth`
- `training/Dockerfile` — CPU PyTorch training image
- `inference/inference.py` — FastAPI server loading saved model
- `inference/Dockerfile` — CPU PyTorch + FastAPI image
- `k8s/pvc.yaml` — PersistentVolumeClaim for model storage
- `k8s/job-training.yaml` — Kubernetes Job for one-time training
- `k8s/deploy-inference.yaml` — Deployment for inference server
- `k8s/svc-inference.yaml` — LoadBalancer Service exposing `/predict`

## Steps + Why (single path)

Use your own values for placeholders `{PROJECT_ID}`, `{REGION}`, `{REPO}`.

1) Prerequisites
```bash
gcloud --version
kubectl version --client
docker --version
export USE_GKE_GCLOUD_AUTH_PLUGIN=True
gcloud components install gke-gcloud-auth-plugin || brew install gke-gcloud-auth-plugin
```

2) Project setup
```bash
export PROJECT_ID={PROJECT_ID}
export REGION={REGION}             # e.g., asia-east1
export REPO={REPO:-mnist-repo}
gcloud auth login
gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION
gcloud services enable container.googleapis.com artifactregistry.googleapis.com
```

3) Artifact Registry (where nodes pull images from)
```bash
gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION || true
gcloud auth configure-docker $REGION-docker.pkg.dev
export TRAIN_IMG=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/mnist-train:latest
export INFER_IMG=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/mnist-infer:latest
```

4) Build and push images (Apple Silicon: build amd64)
```bash
export DOCKER_DEFAULT_PLATFORM=linux/amd64
docker build -t $TRAIN_IMG -f training/Dockerfile . && docker push $TRAIN_IMG
docker build -t $INFER_IMG -f inference/Dockerfile . && docker push $INFER_IMG
```

5) Create GKE Autopilot cluster & credentials
```bash
gcloud container clusters create-auto mnist-cluster --region $REGION
gcloud container clusters get-credentials mnist-cluster --region $REGION --project $PROJECT_ID
```

6) Persist the model with a Job (PVC → Job writes `model.pth`)
```bash
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/job-training.yaml
kubectl logs -l job-name=mnist-training-job -f   # wait for: Saved model to /mnt/model/model.pth
```

7) Serve inference (Deployment) and access it
```bash
kubectl apply -f k8s/deploy-inference.yaml
# Option A: Public URL (incurs LB cost)
kubectl apply -f k8s/svc-inference.yaml
# Option B: Free on-demand
# kubectl port-forward deploy/mnist-inference-deploy 8080:8080
```

8) Test the API
```bash
EXTERNAL_IP=$(kubectl get svc mnist-inference-svc -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl -s http://$EXTERNAL_IP/healthz
python3 samples/gen_samples.py
curl -s -X POST -F "file=@samples/zero.png" http://$EXTERNAL_IP/predict | jq
```

9) Cost control
```bash
kubectl delete svc mnist-inference-svc           # remove LB
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
```

Notes
- PVC is the durable disk; training writes to `/mnt/model`, inference reads from it.
- Apple Silicon must build `linux/amd64` images for GKE nodes.
