#!/usr/bin/env bash
set -e
echo "=== Ensuring Streamlit (venv) and Docker (system) ==="

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJ_DIR/venv_forceps"

if [ ! -d "$VENV" ]; then
  echo "Creating venv at $VENV"
  python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"
pip install --upgrade pip >/dev/null
pip install --quiet streamlit >/dev/null
echo "Streamlit in venv: $(streamlit --version)"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not found. Installing via Homebrew (macOS)..."
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew not found. Installing Homebrew first..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo 'eval "$('/opt/homebrew/bin/brew' shellenv)"' >> "$HOME/.zprofile"
    eval "$('/opt/homebrew/bin/brew' shellenv)"
  fi
  brew install --cask docker
  echo "Docker installed. Please launch Docker.app once to finish setup."
else
  echo "Docker already installed: $(docker --version || true)"
fi

echo "Done."


