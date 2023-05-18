#!/usr/bin/env bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

set -e

PROJECT_ID="asl-fmi"

INSTANCE_NAME="asl-vm"
ZONE="us-central1-a"

BUCKET_NAME="${PROJECT_ID}-asl-bucket"

echo "Setting GCP project $PROJECT_ID for context..."
gcloud config set project $PROJECT_ID

gsutil rm -r gs://"${BUCKET_NAME}"

# remove the -q if you want the confirmation prompt
gcloud compute instances delete -q "${INSTANCE_NAME}" --zone "${ZONE}" --project "${PROJECT_ID}"
