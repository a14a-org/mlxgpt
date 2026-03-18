#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}"
NODE1="${NODE1:-node-1.local}"
SOURCE_ROOT="${CLUSTER_CACHE_ROOT:-${WORKSPACE}/.nanochat-cache}"
TARGET_ROOT="${CLUSTER_TARGET_CACHE_ROOT:-${WORKSPACE}/.nanochat-cache}"

for subdir in tokenizer base_data_climbmix; do
  if [ ! -e "${SOURCE_ROOT}/${subdir}" ]; then
    echo "Missing training input: ${SOURCE_ROOT}/${subdir}" >&2
    exit 1
  fi
done

ssh "${NODE1}" "mkdir -p '${TARGET_ROOT}'"

rsync -az --delete "${SOURCE_ROOT}/tokenizer/" "${NODE1}:${TARGET_ROOT}/tokenizer/"
rsync -az --delete "${SOURCE_ROOT}/base_data_climbmix/" "${NODE1}:${TARGET_ROOT}/base_data_climbmix/"

printf 'Synced training inputs to %s:%s\n' "${NODE1}" "${TARGET_ROOT}"
