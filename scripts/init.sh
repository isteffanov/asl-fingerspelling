#!/usr/bin/env bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

set -e

PROJECT_ID="asl-fmi"

ZONE=n1-standard-4
INSTANCE_NAME="asl-vm"
MACHINE_TYPE="n1-standard-4"
BUCKET_NAME="${PROJECT_ID}-asl-bucket" # bucket names should be globally unique across all GCP projects
RESOURCES_DIRECTORY="../data-generator/data"

# overrides
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case ${key} in
  --resources)
    RESOURCES_DIRECTORY="${2}"
    shift
    shift
    ;;
  *)                   # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift
    ;;
  esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# echo "Setting GCP project $PROJECT_ID for context..."
# gcloud config set project $PROJECT_ID

# create missing resources
if gcloud compute instances describe "${INSTANCE_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" &>/dev/null; then
  echo "Instance ${INSTANCE_NAME} already exists"
else
  echo "Creating VM instance ${INSTANCE_NAME}..."
  gcloud compute instances create "${INSTANCE_NAME}" \
    --machine-type="${MACHINE_TYPE}" \
    --boot-disk-size=10 \
    --boot-disk-type=pd-standard \
    --zone="${ZONE}" \
    --project="${PROJECT_ID}"
fi

if gsutil ls "gs://${BUCKET_NAME}" &>/dev/null; then
  echo "Bucket gs://${BUCKET_NAME} already exists"
else
  echo "Creating GCS bucket ${BUCKET_NAME}..."
  gsutil mb gs://"${BUCKET_NAME}"
fi

# copy missing files to the GCS bucket
# add `-n` to dry-run
# -C: If an error occurs, continue to attempt to copy the remaining files.
# -r: recursive
echo "Copying missing files from ${RESOURCES_DIRECTORY} to the GCS bucket"
gsutil rsync -r -C "${RESOURCES_DIRECTORY}" gs://"${BUCKET_NAME}"
