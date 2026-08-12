#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLD_FILE="${SIM3D_GAZEBO_WORLD:-${ROOT_DIR}/sim3d_gazebo/simple_diff_drive.sdf}"

if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

if [ -f "/opt/ros/humble/setup.bash" ]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  set -u
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

if ! command -v gz >/dev/null 2>&1; then
  echo "Error: gz is not available. Use Dockerfile.sim3d_gazebo or install Gazebo Harmonic."
  exit 1
fi

if ! command -v ros2 >/dev/null 2>&1; then
  echo "Error: ros2 is not available. This demo expects ROS2 Humble."
  exit 1
fi

if [ ! -f "${WORLD_FILE}" ]; then
  echo "Error: world file not found: ${WORLD_FILE}"
  exit 1
fi

cleanup() {
  if [ -n "${BRIDGE_PID:-}" ] && kill -0 "${BRIDGE_PID}" >/dev/null 2>&1; then
    kill "${BRIDGE_PID}" >/dev/null 2>&1 || true
  fi
  if [ -n "${GZ_PID:-}" ] && kill -0 "${GZ_PID}" >/dev/null 2>&1; then
    kill "${GZ_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting Gazebo Harmonic headless server..."
echo "  WORLD=${WORLD_FILE}"
gz sim -s -r "${WORLD_FILE}" &
GZ_PID=$!

echo "Waiting for Gazebo transport to initialize..."
sleep "${SIM3D_GAZEBO_STARTUP_DELAY:-5}"

echo "Starting ROS <-> Gazebo bridge..."
ros2 run ros_gz_bridge parameter_bridge \
  /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
  /odom@nav_msgs/msg/Odometry@gz.msgs.Odometry \
  /tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V &
BRIDGE_PID=$!

echo
echo "Gazebo receiver is running. Open another terminal and run:"
echo "  docker exec -it ${CONTAINER_NAME:-sim3d-gazebo} bash -lc \"source /opt/ros/humble/setup.bash && cd /app && SIM3D_EXECUTE_ROS2=true ./demo_3d_dryrun.sh\""
echo
echo "Optional checks:"
echo "  ros2 topic list"
echo "  python3 /app/src/sim3d_dryrun/watch_gazebo_odom.py --print-period-sec 0.5"
echo "  ros2 topic echo /odom nav_msgs/msg/Odometry --once"
echo "  ros2 run tf2_ros tf2_echo odom base_link"
echo

wait "${GZ_PID}"
