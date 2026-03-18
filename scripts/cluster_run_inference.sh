#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}"
BUILD_DIR="${WORKSPACE}/build/cluster"
PYTHON_BIN="${WORKSPACE}/.venv/bin/python"

BACKEND="${1:-jaccl}"
PROMPT="${2:-The capital of France is}"
MODEL_DIR="${CLUSTER_MODEL_DIR:-${WORKSPACE}/converted/cluster-smoke}"
TOKENIZER_DIR="${CLUSTER_TOKENIZER_DIR:-${MODEL_DIR}/tokenizer}"

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

if [ ! -f "${MODEL_DIR}/config.json" ] || [ ! -f "${MODEL_DIR}/weights.safetensors" ]; then
  echo "Converted model not found at ${MODEL_DIR}" >&2
  exit 1
fi

if [ ! -d "${TOKENIZER_DIR}" ]; then
  echo "Tokenizer directory not found at ${TOKENIZER_DIR}" >&2
  exit 1
fi

OUTPUT_DIR="${BUILD_DIR}/inference-${BACKEND}"
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
  "${WORKSPACE}/scripts/mlx_cluster_inference_smoke.py"
  --backend "${SCRIPT_BACKEND}"
  --model-dir "${MODEL_DIR}"
  --tokenizer-dir "${TOKENIZER_DIR}"
  --prompt "${PROMPT}"
  --max-tokens 16
  --temperature 0.0
  --prepend-bos
  --output-dir "${OUTPUT_DIR}"
)

exec "${CMD[@]}"
