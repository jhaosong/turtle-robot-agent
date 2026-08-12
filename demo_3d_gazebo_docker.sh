#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-rosa-sim3d-gazebo}"
CONTAINER_NAME="${CONTAINER_NAME:-sim3d-gazebo}"
SKIP_BUILD="${SKIP_BUILD:-false}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: Docker is not installed."
  exit 1
fi

if [ "${SKIP_BUILD}" != "true" ]; then
  echo "Building Docker image: ${IMAGE_NAME}"
  docker build -f "${ROOT_DIR}/Dockerfile.sim3d_gazebo" -t "${IMAGE_NAME}" "${ROOT_DIR}"
else
  echo "Skipping Docker build."
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Starting existing container: ${CONTAINER_NAME}"
  docker start -ai "${CONTAINER_NAME}"
else
  ENV_FILE_ARGS=()
  if [ -f "${ROOT_DIR}/.env" ]; then
    ENV_FILE_ARGS+=(--env-file "${ROOT_DIR}/.env")
  fi

  echo "Creating container: ${CONTAINER_NAME}"
  docker run -it \
    --name "${CONTAINER_NAME}" \
    -v "${ROOT_DIR}:/app" \
    "${ENV_FILE_ARGS[@]}" \
    "${IMAGE_NAME}"
fi
