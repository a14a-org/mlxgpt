#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR/nanochat"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12
uv sync --extra cpu

cat <<'EOF'
nanochat is bootstrapped for Apple Silicon / MPS.

Typical next steps:
  cd nanochat
  source .venv/bin/activate
  bash runs/runcpu.sh

For a smaller manual pretrain demo on Apple Silicon:
  python -m scripts.base_train --device-type=mps --depth=4 --head-dim=64 --window-pattern=L \
    --max-seq-len=512 --device-batch-size=8 --total-batch-size=4096 --eval-every=100 \
    --eval-tokens=8192 --core-metric-every=-1 --sample-every=100 --num-iterations=500
EOF
