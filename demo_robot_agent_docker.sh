#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${ROBOT_AGENT_DOCKER_IMAGE:-rosa-robot-agent}"

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not installed or is not on PATH." >&2
    exit 1
fi

if [[ "${SKIP_BUILD:-false}" != "true" ]]; then
    docker build \
        --file "${ROOT_DIR}/Dockerfile.robot_agent" \
        --tag "${IMAGE_NAME}" \
        "${ROOT_DIR}"
elif ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "Error: SKIP_BUILD=true but image '${IMAGE_NAME}' does not exist." >&2
    exit 1
fi

docker_args=(
    run
    --rm
    -it
    --volume "${ROOT_DIR}:/app"
    --workdir /app
    --env PYTHONPATH=/app/src
)

if [[ -f "${ROOT_DIR}/.env" ]]; then
    docker_args+=(--env-file "${ROOT_DIR}/.env")
fi

docker_args+=("${IMAGE_NAME}" python /app/src/robot_agent/cli.py "$@")
docker "${docker_args[@]}"
