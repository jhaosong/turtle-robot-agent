#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TB3_DIR="${ROOT_DIR}/turtlebot3_behavior_demos"
IMAGE_NAME="${ROBOT_AGENT_ROS_IMAGE:-rosa-robot-agent-ros}"
PLATFORM="${ROBOT_AGENT_DOCKER_PLATFORM:-linux/amd64}"
DOCKER_CONFIG_DIR="${ROBOT_AGENT_DOCKER_CONFIG:-${HOME}/.docker-robot-agent}"
GUI_ENABLED="${ROBOT_AGENT_GUI:-true}"

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not installed or is not on PATH." >&2
    exit 1
fi

# Docker Desktop can recreate a WSL config that points at the Windows
# credential helper. This project only pulls public images, so keep its build
# config isolated from Windows login-session availability.
mkdir -p "${DOCKER_CONFIG_DIR}"
if [[ ! -f "${DOCKER_CONFIG_DIR}/config.json" ]]; then
    printf '{}\n' > "${DOCKER_CONFIG_DIR}/config.json"
fi
export DOCKER_CONFIG="${DOCKER_CONFIG_DIR}"

if [[ "${SKIP_BUILD:-false}" != "true" ]]; then
    echo "Building TurtleBot3 ROS2 simulation base..."
    DOCKER_DEFAULT_PLATFORM="${PLATFORM}" docker compose \
        --file "${TB3_DIR}/docker-compose.yaml" \
        --project-directory "${TB3_DIR}" \
        build overlay
    echo "Building ROS2 robot-agent integration image..."
    docker build \
        --platform "${PLATFORM}" \
        --file "${ROOT_DIR}/Dockerfile.robot_agent_ros" \
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
    --name rosa-robot-agent-ros
    --platform "${PLATFORM}"
    --shm-size 1g
    --volume "${ROOT_DIR}:/app"
    --workdir /app
    --env PYTHONPATH=/app/src
    --env ROBOT_AGENT_EXECUTE_ROS2=true
    --env ROBOT_AGENT_ROS_BACKEND=rclpy
    --env ROBOT_AGENT_TOOL_TIMEOUT_SEC="${ROBOT_AGENT_TOOL_TIMEOUT_SEC:-120}"
)

if [[ "${GUI_ENABLED}" == "true" ]]; then
    if [[ ! -d /mnt/wslg ]]; then
        echo "Error: ROBOT_AGENT_GUI=true requires WSLg at /mnt/wslg." >&2
        exit 1
    fi
    : "${DISPLAY:?ROBOT_AGENT_GUI=true requires DISPLAY}"
    docker_args+=(
        --volume /mnt/wslg:/mnt/wslg
        --volume /tmp/.X11-unix:/tmp/.X11-unix
        --env ROBOT_AGENT_GUI=true
        --env DISPLAY="${DISPLAY}"
        --env WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
        --env XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/mnt/wslg/runtime-dir}"
        --env PULSE_SERVER="${PULSE_SERVER:-/mnt/wslg/PulseServer}"
        --env QT_X11_NO_MITSHM=1
    )
    if [[ -n "${LIBGL_ALWAYS_SOFTWARE:-}" ]]; then
        docker_args+=(--env LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE}")
    fi
else
    docker_args+=(--env ROBOT_AGENT_GUI=false)
fi

if [[ -f "${ROOT_DIR}/.env" ]]; then
    docker_args+=(--env-file "${ROOT_DIR}/.env")
fi

# Forward explicitly supplied perception tuning without overriding .env when
# the caller did not set a value in this shell.
for variable_name in \
    ROBOT_AGENT_DETECTOR_BACKEND \
    ROBOT_AGENT_DETECTION_INTERVAL_SEC \
    ROBOT_AGENT_DETECTION_CONFIDENCE_THRESHOLD \
    ROBOT_AGENT_CENTER_ON_DETECTION \
    ROBOT_AGENT_IMAGE_CENTER_TOLERANCE \
    ROBOT_AGENT_CENTERING_MAX_ANGULAR_SPEED \
    ROBOT_AGENT_CENTERING_GAIN \
    ROBOT_AGENT_CENTERING_TIMEOUT_SEC \
    ROBOT_AGENT_YOLO_MODEL \
    ROBOT_AGENT_YOLO_INPUT_SIZE; do
    if [[ -n "${!variable_name:-}" ]]; then
        docker_args+=(--env "${variable_name}=${!variable_name}")
    fi
done

docker_args+=("${IMAGE_NAME}" /app/run_robot_agent_ros.sh)
docker "${docker_args[@]}"
