#!/bin/bash
# Video AI Toolkit - Full Installation Script
# Installs all models and dependencies
# Usage: ./scripts/install.sh

set -e

echo "=== Video AI Toolkit - Full Installation ==="
echo ""

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "[1/6] Creating conda environment (vidtool, Python 3.11)..."
conda create -n vidtool python=3.11 -y

# Activate environment
echo "[2/6] Activating environment..."
eval "$(conda shell.bash hook)"
conda activate vidtool

# Install PyTorch with CUDA
echo "[3/6] Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install this package
echo "[4/6] Installing video-ai-toolkit..."
pip install -e .

# Install model dependencies
echo "[5/6] Installing model dependencies..."
./scripts/install_segment.sh
./scripts/install_depth.sh
./scripts/install_matting.sh
./scripts/install_inpaint.sh

# Verify installation
echo "[6/6] Verifying installation..."
python -c "from video_toolkit import __version__; print(f'video-ai-toolkit v{__version__} installed!')"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "  1. Activate the environment: conda activate vidtool"
echo "  2. Set your HuggingFace token: export HF_TOKEN=hf_xxxx"
echo "  3. Download models: ./scripts/download_models.sh"
echo "  4. Run a command: vidtool segment video.mp4 -c 'person'"
echo ""
