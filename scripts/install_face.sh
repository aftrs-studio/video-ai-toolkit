#!/bin/bash
# Install face restoration models (GFPGAN, CodeFormer)
set -e

echo "Installing face restoration dependencies..."

pip install gfpgan

echo ""
echo "Downloading GFPGAN models..."
mkdir -p ~/.cache/vidtool/gfpgan

if [ ! -f ~/.cache/vidtool/gfpgan/GFPGANv1.4.pth ]; then
    echo "Downloading GFPGANv1.4..."
    wget -q -P ~/.cache/vidtool/gfpgan/ "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" 2>/dev/null || echo "Note: Model may need manual download"
fi

echo ""
echo "Setting up CodeFormer..."
mkdir -p ~/.cache/vidtool/codeformer

if [ ! -d ~/.cache/vidtool/codeformer/CodeFormer ]; then
    echo "Cloning CodeFormer repository..."
    git clone https://github.com/sczhou/CodeFormer ~/.cache/vidtool/codeformer/CodeFormer
    cd ~/.cache/vidtool/codeformer/CodeFormer && pip install -r requirements.txt
fi

echo ""
echo "Face restoration models installed!"
echo "Usage: vidtool face video.mp4 -m gfpgan"
