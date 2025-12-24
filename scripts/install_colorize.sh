#!/bin/bash
# Install video colorization models (DeOldify)
set -e

echo "Installing colorization dependencies..."

pip install torch torchvision fastai opencv-python numpy Pillow

echo ""
echo "Installing DeOldify..."
pip install deoldify || echo "Note: DeOldify may need manual installation from https://github.com/jantic/DeOldify"

echo ""
echo "Downloading DeOldify models..."
mkdir -p ~/.cache/vidtool/deoldify

MODELS=("ColorizeVideo_gen" "ColorizeArtistic_gen")
for model in "${MODELS[@]}"; do
    if [ ! -f ~/.cache/vidtool/deoldify/${model}.pth ]; then
        echo "Downloading ${model}..."
        wget -q -P ~/.cache/vidtool/deoldify/ "https://data.deepai.org/deoldify/${model}.pth" 2>/dev/null || echo "Note: ${model} may need manual download"
    fi
done

echo ""
echo "Colorization models installed!"
echo "Usage: vidtool colorize bw_footage.mp4"
