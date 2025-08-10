#!/usr/bin/env bash
set -e
echo "=== FORCEPS Installer ==="

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PYTHON:-python3}
VENV="$ROOT_DIR/venv_forceps"

if [ ! -d "$VENV" ]; then
  "$PY" -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r "$ROOT_DIR/app/requirements.txt"

echo "Installation complete."
echo "Activate environment and run:"
echo "  source $VENV/bin/activate"
echo "  PYTHONPATH=$ROOT_DIR streamlit run $ROOT_DIR/app/main.py"
