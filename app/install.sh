#!/usr/bin/env bash
set -e
echo "=== FORCEPS Installer ==="

echo "Installing dependencies for the current user."
echo "Step 1: Installing CPU-only PyTorch to save space..."

python3 -m pip install --user --upgrade pip
python3 -m pip install --user --index-url https://download.pytorch.org/whl/cpu torch torchvision

echo "Step 2: Installing remaining dependencies..."
python3 -m pip install --user -r "app/requirements.txt"

echo "Installation complete."
echo "Dependencies installed in the user's site-packages."
echo "Ensure ~/.local/bin is in your PATH to run installed scripts."
