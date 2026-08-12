#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export SIM3D_TRACE="${SIM3D_TRACE:-true}"

python3 "${ROOT_DIR}/src/sim3d_dryrun/agent.py"
