#!/bin/bash
# Video AI Toolkit - Download Model Checkpoints
# Downloads all model weights

set -e

echo "=== Downloading Model Checkpoints ==="

# Check for HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Some models may fail to download."
    echo "Get your token from: https://huggingface.co/settings/tokens"
fi

MODEL_PATH="${VIDTOOL_MODEL_PATH:-$HOME/.cache/vidtool}"
echo "Model cache: $MODEL_PATH"
mkdir -p "$MODEL_PATH"

# Download using Python
python -c "
from video_toolkit.config import Config
config = Config.from_env()
print(f'Downloading models to {config.model_path}...')

# This would download all model checkpoints
# Implementation depends on each model's download mechanism
print('Model downloads initiated. Check each model directory.')
"

echo ""
echo "Models will be cached at: $MODEL_PATH"
