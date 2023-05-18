#!/bin/bash

PROJECT_ID="asl-fmi"
ZONE="us-central1-a"
INSTANCE_NAME="asl-vm"
BUCKET_NAME="${PROJECT_ID}-asl-bucket"
RESOURCES_DIRECTORY=""
CMD="start"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case ${key} in
  --resources)
    RESOURCES_DIRECTORY="${2}"
    shift
    shift
    ;;
  --start)
    CMD="start"
    shift
    ;;
  --stop)
    CMD="stop"
    shift
    ;;
  *)
    echo "Unknown flag ${1}"
    exit 1
    ;;
  esac
done

is_running() {
  gcloud compute instances describe "$INSTANCE_NAME" --zone "$ZONE" --project="${PROJECT_ID}" --format='value(status)' | grep -q "RUNNING"
}

start() {
  echo "start"
  if [ $RESOURCES_DIRECTORY ]; then
    echo "Copying missing resource files from resources directory"
    gsutil -m rsync -r -x ".*" -C "${RESOURCES_DIRECTORY}" gs://"${BUCKET_NAME}"
  fi

  echo "Starting instance $INSTANCE_NAME..."
  if is_running; then
    echo "Instance $INSTANCE_NAME is already running"
    return
  fi

  gcloud compute instances start "$INSTANCE_NAME" --zone "$ZONE" --project="${PROJECT_ID}"
}

stop() {
  echo "Stopping instance $INSTANCE_NAME..."
  if is_running; then
    gcloud compute instances stop "$INSTANCE_NAME" --zone "$ZONE" --project="${PROJECT_ID}"
    return
  fi

  echo "Instance $INSTANCE_NAME is already stopped"
}

if [ "$CMD" == "start" ]; then
  start
elif [ "$CMD" == "stop" ]; then
  stop
else
  echo "Invalid argument. Usage: run-vm.sh --start | --stop"
  exit 1
fi
