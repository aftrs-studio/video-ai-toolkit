#!/bin/bash
# Install optical flow models (RAFT, UniMatch)
set -e

echo "Installing optical flow dependencies..."

pip install torch torchvision

echo ""
echo "Setting up RAFT..."
mkdir -p ~/.cache/vidtool/raft

if [ ! -d ~/.cache/vidtool/raft/RAFT ]; then
    echo "Cloning RAFT repository..."
    git clone https://github.com/princeton-vl/RAFT ~/.cache/vidtool/raft/RAFT
fi

echo "Downloading RAFT models..."
cd ~/.cache/vidtool/raft/RAFT
if [ ! -f models/raft-things.pth ]; then
    ./download_models.sh 2>/dev/null || echo "Note: Models may need manual download"
fi

echo ""
echo "Setting up UniMatch..."
mkdir -p ~/.cache/vidtool/unimatch

if [ ! -d ~/.cache/vidtool/unimatch/unimatch ]; then
    echo "Cloning UniMatch repository..."
    git clone https://github.com/autonomousvision/unimatch ~/.cache/vidtool/unimatch/unimatch
fi

echo ""
echo "Optical flow models installed!"
echo "Usage: vidtool flow video.mp4 -m raft"
