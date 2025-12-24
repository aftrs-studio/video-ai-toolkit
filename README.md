# Video AI Toolkit

Unified Python toolkit for video processing with state-of-the-art AI models. Segment objects, estimate depth, remove backgrounds, and inpaint videos with a single CLI.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/aftrs-studio/video-ai-toolkit.git && cd video-ai-toolkit && ./scripts/install.sh

# 2. Activate environment
conda activate vidtool

# 3. Set HuggingFace token
export HF_TOKEN=hf_xxxxxxxxxxxx

# 4. Process videos!
vidtool segment video.mp4 -c "person,car"
vidtool depth video.mp4
vidtool matte video.mp4
vidtool inpaint video.mp4 --mask mask.mp4
```

---

## Available Models

### Segmentation & Tracking
| Model | Command | Description |
|-------|---------|-------------|
| **Grounded SAM 2** | `-m grounded` | Text-prompted segmentation (default) |
| **SAM 2** | `-m sam2` | Point/box prompted segmentation |
| **DEVA** | `-m deva` | Decoupled video segmentation |

### Depth Estimation
| Model | Command | Description |
|-------|---------|-------------|
| **Video Depth Anything** | `-m video-depth` | Consistent depth for long videos (default) |
| **Depth Anything V2** | `-m depth-v2` | Fast monocular depth |

### Background Removal (Matting)
| Model | Command | Description |
|-------|---------|-------------|
| **RobustVideoMatting** | `-m rvm` | Real-time matting, 4K@76fps (default) |
| **MODNet** | `-m modnet` | Portrait-optimized matting |

### Object Removal (Inpainting)
| Model | Command | Description |
|-------|---------|-------------|
| **ProPainter** | `-m propainter` | High-quality inpainting (default) |
| **E2FGVI** | `-m e2fgvi` | Flow-guided inpainting |

---

## Commands

### Segment Objects

```bash
# Segment people and cars with text prompts
vidtool segment video.mp4 -c "person,car"

# Use specific model
vidtool segment video.mp4 -c "person" -m deva

# Detailed prompts work too
vidtool segment video.mp4 -c "person wearing red shirt"
```

### Estimate Depth

```bash
# Default: Video Depth Anything (large)
vidtool depth video.mp4

# Faster: Depth V2 with small variant
vidtool depth video.mp4 -m depth-v2 -v small
```

### Remove Background

```bash
# Default: RobustVideoMatting with green background
vidtool matte video.mp4

# Blue background
vidtool matte video.mp4 -b "0,0,255"

# Portrait-optimized
vidtool matte video.mp4 -m modnet
```

### Remove Objects

```bash
# Inpaint using mask video
vidtool inpaint video.mp4 --mask mask.mp4

# Alternative model
vidtool inpaint video.mp4 --mask mask.mp4 -m e2fgvi
```

### System Info

```bash
# Show GPU and system info
vidtool info

# List all available models
vidtool models
```

---

## Installation

### Full Installation (All Models)

```bash
./scripts/install.sh
```

### Selective Installation

Install only the models you need:

```bash
# Core package only
pip install -e .

# Then install specific models
./scripts/install_segment.sh   # SAM 2, Grounded SAM 2, DEVA
./scripts/install_depth.sh     # Video Depth Anything, Depth V2
./scripts/install_matting.sh   # RVM, MODNet
./scripts/install_inpaint.sh   # ProPainter, E2FGVI
```

---

## Output

All commands output to `./output/` by default:

```
output/
├── video_segment_person_001.mp4   # Segmented object
├── video_depth_large.mp4          # Depth visualization
├── video_matte_rvm.mp4            # Background removed
├── video_inpaint_propainter.mp4   # Object removed
└── video_metadata.json            # Processing info
```

---

## Requirements

- Python 3.11+
- PyTorch 2.3+ with CUDA
- NVIDIA GPU with 8GB+ VRAM (16GB recommended)
- ~20GB disk for all model checkpoints

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token | (required) |
| `VIDTOOL_MODEL_PATH` | Model cache | `~/.cache/vidtool` |
| `VIDTOOL_DEVICE` | cuda or cpu | `cuda` |
| `VIDTOOL_OUTPUT_DIR` | Output directory | `./output` |

---

## Model Selection Guide

| Task | Best For | Alternative |
|------|----------|-------------|
| Segment by text | Grounded SAM 2 | DEVA (faster) |
| Track objects | SAM 2 | DEVA |
| Depth (quality) | Video Depth Anything | - |
| Depth (speed) | Depth V2 small | - |
| Full body matting | RobustVideoMatting | - |
| Portrait matting | MODNet | RVM |
| Object removal | ProPainter | E2FGVI (faster) |

---

## AFTRS Studio Repositories

| Repository | Description |
|------------|-------------|
| [aftrs-mcp](https://github.com/aftrs-studio/aftrs-mcp) | MCP server with 177+ tools for TouchDesigner, lighting, Resolume, OBS |
| [sam3-video-segmenter](https://github.com/aftrs-studio/sam3-video-segmenter) | SAM 3 video segmentation tool |
| [video-ai-toolkit](https://github.com/aftrs-studio/video-ai-toolkit) | This repo - unified video AI toolkit |

---

## License

MIT License - see LICENSE file.

## Credits

- [SAM 2](https://github.com/facebookresearch/sam2) - Meta AI
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) - IDEA Research
- [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA) - hkchengrex
- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) - DepthAnything
- [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) - Peter Lin
- [ProPainter](https://github.com/sczhou/ProPainter) - Shangchen Zhou
