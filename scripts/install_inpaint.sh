#!/bin/bash
# Video AI Toolkit - Install Inpainting Models
# Installs ProPainter, E2FGVI

set -e

echo "=== Installing Inpainting Models ==="

# ProPainter
echo "[1/2] Installing ProPainter..."
pip install git+https://github.com/sczhou/ProPainter.git

# E2FGVI
echo "[2/2] Installing E2FGVI..."
pip install git+https://github.com/MCG-NKU/E2FGVI.git

echo ""
echo "Inpainting models installed!"
echo "Available: propainter, e2fgvi"
