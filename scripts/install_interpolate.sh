#!/bin/bash
# Install frame interpolation models (RIFE, FILM)
set -e

echo "Installing frame interpolation dependencies..."

pip install torch torchvision

echo ""
echo "Setting up RIFE..."
mkdir -p ~/.cache/vidtool/rife

if [ ! -d ~/.cache/vidtool/rife/ECCV2022-RIFE ]; then
    echo "Cloning RIFE repository..."
    git clone https://github.com/hzwer/ECCV2022-RIFE ~/.cache/vidtool/rife/ECCV2022-RIFE
fi

echo ""
echo "Setting up FILM..."
pip install tensorflow tensorflow-hub

mkdir -p ~/.cache/vidtool/film
echo "FILM models will be downloaded on first use from TensorFlow Hub."

echo ""
echo "Frame interpolation models installed!"
echo "Usage: vidtool interpolate video.mp4 --multiplier 2"
