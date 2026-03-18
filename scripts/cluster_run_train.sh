#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}"
BUILD_DIR="${WORKSPACE}/build/cluster"
PYTHON_BIN="${WORKSPACE}/.venv/bin/python"

BACKEND="${1:-jaccl}"
PARALLELISM="${2:-dp}"
shift $(( $# >= 1 ? 1 : 0 ))
shift $(( $# >= 1 ? 1 : 0 ))

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

case "$PARALLELISM" in
  dp|tp) ;;
  *)
    echo "Unsupported parallelism: ${PARALLELISM}" >&2
    echo "Use one of: dp, tp" >&2
    exit 1
    ;;
esac

"${WORKSPACE}/scripts/cluster_write_hostfiles.sh"

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
  "${WORKSPACE}/scripts/train_mlx_cluster.py"
  --backend "${SCRIPT_BACKEND}"
  --parallelism "${PARALLELISM}"
)

if [ "$#" -gt 0 ]; then
  CMD+=("$@")
fi

exec "${CMD[@]}"
