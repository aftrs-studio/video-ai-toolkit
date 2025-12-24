#!/bin/bash
# Video AI Toolkit - Install Segmentation Models
# Installs SAM 2, Grounded SAM 2, DEVA

set -e

echo "=== Installing Segmentation Models ==="

# SAM 2
echo "[1/3] Installing SAM 2..."
pip install git+https://github.com/facebookresearch/sam2.git

# Grounding DINO (for Grounded SAM 2)
echo "[2/3] Installing Grounding DINO..."
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

# DEVA
echo "[3/3] Installing DEVA..."
pip install git+https://github.com/hkchengrex/Tracking-Anything-with-DEVA.git

echo ""
echo "Segmentation models installed!"
echo "Available: sam2, grounded, deva"
