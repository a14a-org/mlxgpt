#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}"
BUILD_DIR="${WORKSPACE}/build/cluster"
PYTHON_BIN="${WORKSPACE}/.venv/bin/python"

BACKEND="${1:-ring}"
case "$BACKEND" in
  ring)
    LAUNCH_BACKEND="ring"
    SCRIPT_BACKEND="ring"
    EXTRA_ENV=()
    HOSTFILE="${BUILD_DIR}/ring-2.json"
    ;;
  jaccl)
    LAUNCH_BACKEND="jaccl"
    SCRIPT_BACKEND="jaccl"
    EXTRA_ENV=(--env MLX_METAL_FAST_SYNCH=1)
    HOSTFILE="${BUILD_DIR}/jaccl-2.json"
    ;;
  jaccl-ring)
    LAUNCH_BACKEND="jaccl-ring"
    SCRIPT_BACKEND="any"
    EXTRA_ENV=(--env MLX_METAL_FAST_SYNCH=1)
    HOSTFILE="${BUILD_DIR}/jaccl-2.json"
    ;;
  *)
    echo "Unsupported backend: ${BACKEND}" >&2
    echo "Use one of: ring, jaccl, jaccl-ring" >&2
    exit 1
    ;;
esac

"${WORKSPACE}/scripts/cluster_write_hostfiles.sh"

OUTPUT_DIR="${BUILD_DIR}/smoke-${BACKEND}"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

CMD=(
  "${WORKSPACE}/.venv/bin/mlx.launch"
  --verbose
  --backend "${LAUNCH_BACKEND}"
  --hostfile "${HOSTFILE}"
  --python "${PYTHON_BIN}"
  --cwd "${WORKSPACE}"
)

if [ "${#EXTRA_ENV[@]}" -gt 0 ]; then
  CMD+=("${EXTRA_ENV[@]}")
fi

CMD+=(
  --
  "${WORKSPACE}/scripts/mlx_distributed_smoke.py"
  --backend "${SCRIPT_BACKEND}"
  --output-dir "${OUTPUT_DIR}"
)

exec "${CMD[@]}"
