#!/bin/bash
# Video AI Toolkit - Install Matting Models
# Installs RobustVideoMatting, MODNet

set -e

echo "=== Installing Matting Models ==="

# RobustVideoMatting (available via torch.hub)
echo "[1/2] RobustVideoMatting available via torch.hub..."

# MODNet
echo "[2/2] Installing MODNet..."
pip install git+https://github.com/ZHKKKe/MODNet.git

echo ""
echo "Matting models installed!"
echo "Available: rvm, modnet"
