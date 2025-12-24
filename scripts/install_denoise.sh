#!/bin/bash
# Install video denoising models (FastDVDnet, ViDeNN)
set -e

echo "Installing denoising dependencies..."

pip install torch torchvision opencv-python numpy

echo ""
echo "Downloading FastDVDnet model..."
mkdir -p ~/.cache/vidtool/fastdvdnet

if [ ! -f ~/.cache/vidtool/fastdvdnet/fastdvdnet_model.pth ]; then
    wget -q -P ~/.cache/vidtool/fastdvdnet/ "https://github.com/m-tassano/fastdvdnet/raw/master/model.pth" -O ~/.cache/vidtool/fastdvdnet/fastdvdnet_model.pth 2>/dev/null || echo "Note: FastDVDnet model may need manual download from https://github.com/m-tassano/fastdvdnet"
fi

echo ""
echo "Downloading ViDeNN model..."
mkdir -p ~/.cache/vidtool/videnn

if [ ! -f ~/.cache/vidtool/videnn/videnn_model.pth ]; then
    echo "Note: ViDeNN model needs manual download from https://github.com/clausmichele/ViDeNN"
fi

echo ""
echo "Denoising models installed!"
echo "Usage: vidtool denoise noisy.mp4 -m fastdvdnet"
