#!/bin/bash
# Install upscaling models (Real-ESRGAN, Video2X)
set -e

echo "Installing upscaling dependencies..."

pip install realesrgan basicsr

echo ""
echo "Downloading Real-ESRGAN models..."
mkdir -p ~/.cache/vidtool/realesrgan

MODELS=("RealESRGAN_x4plus" "RealESRGAN_x4plus_anime_6B" "realesr-animevideov3")
for model in "${MODELS[@]}"; do
    if [ ! -f ~/.cache/vidtool/realesrgan/${model}.pth ]; then
        echo "Downloading ${model}..."
        wget -q -P ~/.cache/vidtool/realesrgan/ "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/${model}.pth" 2>/dev/null || echo "Note: ${model} may need manual download"
    fi
done

echo ""
echo "Upscaling models installed!"
echo "Usage: vidtool upscale video.mp4 --scale 4"
