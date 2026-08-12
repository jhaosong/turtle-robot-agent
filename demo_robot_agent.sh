#!/usr/bin/env bash
set -euo pipefail

ROBOT_AGENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${ROBOT_AGENT_ROOT}/.env" ]]; then
  set -a
  source "${ROBOT_AGENT_ROOT}/.env"
  set +a
fi

if [[ -f /opt/ros/humble/setup.bash ]]; then
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi

export PYTHONPATH="${ROBOT_AGENT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
python3 "${ROBOT_AGENT_ROOT}/src/robot_agent/cli.py" "$@"
