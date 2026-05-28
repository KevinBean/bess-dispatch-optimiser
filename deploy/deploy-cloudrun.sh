#!/usr/bin/env bash
# Deploy the BESS optimiser demo to Google Cloud Run (free tier, scales to zero).
#
# Why Cloud Run: no local Docker needed — Cloud Build builds the image remotely from
# source. With min-instances=0 the service costs nothing when idle and stays inside
# the Cloud Run free tier (2M requests, 360k vCPU-sec, 180k GiB-sec / month).
#
# One-time prereqs:
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID      # project must have billing enabled
#   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
#
# Then, from the repo root:
#   bash deploy/deploy-cloudrun.sh
#
# The demo deploys WITHOUT an OpenAI key (agent tab shows a notice; optimiser +
# forecast + money chart run live). To enable the agent later, store a key and
# redeploy with the SET-SECRET block at the bottom uncommented.
set -euo pipefail

REGION="${REGION:-australia-southeast1}"     # Sydney
SERVICE="${SERVICE:-bess-dispatch-optimiser}"
PROJECT="$(gcloud config get-value project 2>/dev/null)"

if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
  echo "No gcloud project set. Run: gcloud config set project YOUR_PROJECT_ID" >&2
  exit 1
fi
echo "==> Deploying '$SERVICE' to project '$PROJECT' in '$REGION' (build from source)"

# Build from source (Cloud Build) + deploy. --source uses the repo's Dockerfile.
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --cpu-boost \
  --min-instances 0 \
  --max-instances 2 \
  --timeout 300 \
  --port 8080 \
  --set-env-vars ANONYMIZED_TELEMETRY=False,HF_HUB_DISABLE_TELEMETRY=1

echo
echo "==> Live URL:"
gcloud run services describe "$SERVICE" --region "$REGION" \
  --format 'value(status.url)'

# --- OPTIONAL: enable the LLM agent later -----------------------------------
# echo -n "sk-..." | gcloud secrets create bess-openai-key --data-file=- 2>/dev/null || \
#   echo -n "sk-..." | gcloud secrets versions add bess-openai-key --data-file=-
# gcloud run services update "$SERVICE" --region "$REGION" \
#   --set-secrets OPENAI_API_KEY=bess-openai-key:latest
