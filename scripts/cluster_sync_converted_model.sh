#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL_NAME="${1:-cluster-smoke}"
NODE0="${NODE0:-node-0.local}"
NODE1="${NODE1:-node-1.local}"
SOURCE_DIR="${CLUSTER_SOURCE_MODEL_DIR:-${WORKSPACE}/converted/${MODEL_NAME}}"
TARGET_DIR="${CLUSTER_TARGET_MODEL_DIR:-${WORKSPACE}/converted/${MODEL_NAME}}"

if [ ! -f "${SOURCE_DIR}/config.json" ] || [ ! -f "${SOURCE_DIR}/weights.safetensors" ]; then
  echo "Converted model not found at ${SOURCE_DIR}" >&2
  exit 1
fi

ssh "${NODE1}" "mkdir -p '${TARGET_DIR}'"

rsync -az --delete \
  "${SOURCE_DIR}/" \
  "${NODE1}:${TARGET_DIR}/"

printf 'Synced %s -> %s:%s\n' "${SOURCE_DIR}" "${NODE1}" "${TARGET_DIR}"
