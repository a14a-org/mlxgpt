#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x /opt/homebrew/bin/python3 ]; then
    PYTHON_BIN=/opt/homebrew/bin/python3
  else
    PYTHON_BIN=python3
  fi
fi

if [ -d .venv ]; then
  VENV_PYTHON="$(./.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  case "$VENV_PYTHON" in
    3.11|3.12|3.13|3.14) ;;
    *)
      rm -rf .venv
      ;;
  esac
fi

"$PYTHON_BIN" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .

cat <<'EOF'
MLX environment is ready.

Next steps:
  . .venv/bin/activate
  python scripts/smoke_test_mlx.py
EOF
