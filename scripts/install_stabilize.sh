#!/bin/bash
# Install video stabilization dependencies
set -e

echo "Installing stabilization dependencies..."

pip install opencv-python numpy

echo ""
echo "Stabilization installed!"
echo "Uses optical flow-based stabilization (no additional models needed)"
echo "Usage: vidtool stabilize shaky.mp4"
