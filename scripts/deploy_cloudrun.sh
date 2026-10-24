#!/bin/bash
# Builds the demo image on Cloud Build and deploys it to Cloud Run with one L4 GPU.
# One-time project setup first: docs/deploy.md.
#
#   PROJECT=my-gcp-project bash scripts/deploy_cloudrun.sh
#   PROJECT=my-gcp-project REGION=europe-west1 bash scripts/deploy_cloudrun.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT=${PROJECT:?set PROJECT to your GCP project id}
# L4 on Cloud Run is offered in a subset of regions; us-central1 is one of them.
REGION=${REGION:-us-central1}
SERVICE=${SERVICE:-murmur}
REPO=${REPO:-murmur}
# Haiku: ~900 ms first token through OpenRouter against Opus's ~2.9 s, at ~1/40 the price.
MODEL=${MODEL:-anthropic/claude-haiku-5.5}
# GPU=0 deploys CPU-only: no GPU quota needed, slower (base.en Whisper, Kokoro on CPU).
GPU=${GPU:-1}
SKIP_BUILD=${SKIP_BUILD:-}

if [ "$GPU" = "1" ]; then
  HARDWARE=(--gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy --cpu 4 --memory 16Gi)
  DEVICE_ENV="MURMUR_TTS_DEVICE=cuda,MURMUR_STT_DEVICE=cuda"
else
  HARDWARE=(--cpu 4 --memory 8Gi)
  # Partials re-transcribe the utterance; on 4 cores they would starve the final pass.
  DEVICE_ENV="MURMUR_TTS_DEVICE=cpu,MURMUR_STT_DEVICE=cpu,MURMUR_STT_COMPUTE_TYPE=int8"
  DEVICE_ENV="$DEVICE_ENV,MURMUR_STT_MODEL=base.en,MURMUR_STT_PARTIAL_INTERVAL_MS=0"
fi
TAG=$(git rev-parse --short HEAD)
IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO/murmur:$TAG"

if [ -n "$(git status --porcelain -- src scripts Dockerfile pyproject.toml uv.lock)" ]; then
  echo "Uncommitted changes in what the image contains; commit first so $TAG means something." >&2
  exit 1
fi

if [ -z "$SKIP_BUILD" ]; then
  echo "== building $IMAGE on Cloud Build (uploads source only; models download on Google's side)"
  gcloud builds submit . \
    --project "$PROJECT" --region "$REGION" \
    --tag "$IMAGE" \
    --machine-type e2-highcpu-8 \
    --timeout 3600s
fi

echo "== deploying $SERVICE"
# --max-instances 1: one GPU, and the app serves one conversation at a time anyway.
# --min-instances 0: scale to zero when idle, so an unused demo costs nothing; the first
#   visitor after idle waits for a cold start while the models load.
# --no-cpu-throttling: VAD, partial transcription and the memory worker run between requests.
# --timeout 3600: a conversation is one WebSocket; Cloud Run's maximum request length.
# --session-affinity: page, memory panel and WebSocket land on the same instance.
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" --region "$REGION" \
  --image "$IMAGE" \
  "${HARDWARE[@]}" --no-cpu-throttling \
  --min-instances 0 --max-instances 1 --concurrency 10 \
  --timeout 3600 --session-affinity \
  --set-env-vars "MURMUR_HOSTED_MODEL=$MODEL,$DEVICE_ENV" \
  --set-secrets "MURMUR_OPENROUTER_API_KEY=openrouter-api-key:latest" \
  --startup-probe "httpGet.path=/healthz,periodSeconds=10,failureThreshold=30,timeoutSeconds=5" \
  --allow-unauthenticated

URL=$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --format 'value(status.url)')
echo "== live: $URL"
