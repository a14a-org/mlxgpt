#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}"
BUILD_DIR="${WORKSPACE}/build/cluster"

NODE0="${NODE0:-node-0.local}"
NODE1="${NODE1:-node-1.local}"
NODE0_IP="${NODE0_IP:-192.168.178.56}"
NODE1_IP="${NODE1_IP:-192.168.178.159}"
RDMA_DEV_NODE0="${RDMA_DEV_NODE0:-rdma_en4}"
RDMA_DEV_NODE1="${RDMA_DEV_NODE1:-rdma_en2}"

mkdir -p "$BUILD_DIR"

cat > "${BUILD_DIR}/ring-2.json" <<EOF
[
  {
    "ssh": "${NODE0}",
    "ips": ["${NODE0_IP}"]
  },
  {
    "ssh": "${NODE1}",
    "ips": ["${NODE1_IP}"]
  }
]
EOF

cat > "${BUILD_DIR}/jaccl-2.json" <<EOF
{
  "backend": "jaccl",
  "envs": [],
  "hosts": [
    {
      "ssh": "${NODE0}",
      "ips": ["${NODE0_IP}"],
      "rdma": [null, "${RDMA_DEV_NODE0}"]
    },
    {
      "ssh": "${NODE1}",
      "ips": ["${NODE1_IP}"],
      "rdma": ["${RDMA_DEV_NODE1}", null]
    }
  ]
}
EOF

printf 'Wrote %s\n' "${BUILD_DIR}/ring-2.json"
printf 'Wrote %s\n' "${BUILD_DIR}/jaccl-2.json"
