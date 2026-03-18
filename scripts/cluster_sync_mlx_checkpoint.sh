#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}"
NODE1="${NODE1:-node-1.local}"
CHECKPOINT_ROOT="${CLUSTER_MLX_CHECKPOINT_ROOT:-${WORKSPACE}/.mlx-checkpoints}"

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <model-tag>" >&2
  exit 1
fi

MODEL_TAG="$1"
SOURCE_DIR="${CHECKPOINT_ROOT}/${MODEL_TAG}"

if [ ! -d "${SOURCE_DIR}" ]; then
  echo "Missing checkpoint root: ${SOURCE_DIR}" >&2
  exit 1
fi

ssh "${NODE1}" "mkdir -p '${CHECKPOINT_ROOT}'"
rsync -az --delete "${SOURCE_DIR}/" "${NODE1}:${SOURCE_DIR}/"

printf 'Synced MLX checkpoints to %s:%s\n' "${NODE1}" "${SOURCE_DIR}"
