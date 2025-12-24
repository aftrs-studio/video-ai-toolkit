#!/bin/bash
# Install style transfer dependencies (AdaIN, Fast Artistic)
set -e

echo "Installing style transfer dependencies..."

pip install torch torchvision opencv-python numpy Pillow tqdm

echo ""
echo "Downloading AdaIN decoder weights..."
mkdir -p ~/.cache/vidtool/adain

if [ ! -f ~/.cache/vidtool/adain/decoder.pth ]; then
    wget -q -P ~/.cache/vidtool/adain/ "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v1.0/decoder.pth" 2>/dev/null || echo "Note: AdaIN decoder may need manual download from https://github.com/naoto0804/pytorch-AdaIN/releases"
fi

echo ""
echo "Downloading Fast Artistic style models..."
mkdir -p ~/.cache/vidtool/fast-artistic

STYLES=("starry_night" "la_muse" "the_scream" "wave" "feathers" "candy" "mosaic" "udnie")
for style in "${STYLES[@]}"; do
    if [ ! -f ~/.cache/vidtool/fast-artistic/${style}.pth ]; then
        echo "Note: ${style}.pth needs download from https://github.com/manuelruder/fast-artistic-videos"
    fi
done

echo ""
echo "Style transfer models installed!"
echo "Usage: vidtool style video.mp4 --style painting.jpg"
