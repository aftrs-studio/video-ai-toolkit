# Video AI Toolkit - Claude Code Instructions

## Project Overview
Unified Python toolkit with multiple AI models for video processing: segmentation, tracking, depth estimation, matting (background removal), and inpainting (object removal).

## Tech Stack
- Python 3.11+
- PyTorch 2.3+ with CUDA 12.x
- Click for CLI
- OpenCV + ffmpeg for video I/O

## Available Models

| Category | Models |
|----------|--------|
| Segmentation | SAM 2, Grounded SAM 2, DEVA |
| Depth | Video Depth Anything, Depth Anything V2 |
| Matting | RobustVideoMatting, MODNet |
| Inpainting | ProPainter, E2FGVI |

## Project Structure
```
video-ai-toolkit/
├── video_toolkit/           # Main Python package
│   ├── cli.py               # Unified CLI
│   ├── config.py            # Configuration
│   ├── video_io.py          # Shared video I/O
│   ├── utils.py             # Shared utilities
│   ├── segment/             # Segmentation tools
│   ├── depth/               # Depth estimation
│   ├── matting/             # Background removal
│   └── inpaint/             # Object removal
├── scripts/                 # Installation scripts
├── examples/                # Usage examples
└── docs/                    # Documentation
```

## CLI Commands
```bash
vidtool segment video.mp4 -c "person"     # Segment objects
vidtool depth video.mp4                    # Estimate depth
vidtool matte video.mp4                    # Remove background
vidtool inpaint video.mp4 --mask mask.mp4  # Remove objects
vidtool info                               # System info
vidtool models                             # List models
```

## Code Standards

### Python Style
- Use type hints everywhere
- Format with Black (line length 88)
- Lint with Ruff
- Docstrings for public functions

### Model Wrapper Pattern
```python
class ModelWrapper:
    def __init__(self, config: Config):
        self.config = config
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        # Load model here

    def process(self, video_path: Path) -> Result:
        self._load_model()
        # Process video
```

## Environment Variables
- `HF_TOKEN` - HuggingFace API token
- `VIDTOOL_MODEL_PATH` - Model cache (default: ~/.cache/vidtool)
- `VIDTOOL_DEVICE` - cuda or cpu (default: cuda)
- `VIDTOOL_OUTPUT_DIR` - Output directory (default: ./output)

## Important Notes
- Models have varying VRAM requirements (8-16GB)
- Use modular install scripts for selective installation
- Each model wrapper follows the same interface pattern
