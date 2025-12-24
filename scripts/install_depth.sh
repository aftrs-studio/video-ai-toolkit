#!/bin/bash
# Video AI Toolkit - Install Depth Estimation Models
# Installs Video Depth Anything, Depth Anything V2

set -e

echo "=== Installing Depth Estimation Models ==="

# Video Depth Anything
echo "[1/2] Installing Video Depth Anything..."
pip install git+https://github.com/DepthAnything/Video-Depth-Anything.git

# Depth Anything V2
echo "[2/2] Installing Depth Anything V2..."
pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git

echo ""
echo "Depth models installed!"
echo "Available: video-depth, depth-v2"
