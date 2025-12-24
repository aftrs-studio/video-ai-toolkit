#!/bin/bash
# Install video generation models (Wan 2.1, CogVideoX)
set -e

echo "Installing video generation dependencies..."

pip install torch torchvision diffusers transformers accelerate sentencepiece

echo ""
echo "Video generation models will be downloaded on first use from HuggingFace."
echo ""
echo "Available models:"
echo "  - Wan 2.1 (Alibaba): T2V-1.3B, T2V-14B, I2V-14B"
echo "  - CogVideoX (Zhipu): 2B, 5B, 5B-I2V"
echo ""
echo "Note: These models require significant VRAM (8-16GB+)"
echo "Set HF_TOKEN environment variable for gated models."
echo ""
echo "Usage: vidtool generate -p 'a cat walking on the beach'"
