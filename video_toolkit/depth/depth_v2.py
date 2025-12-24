"""Depth Anything V2 - Fast monocular depth estimation."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from video_toolkit.config import Config
from video_toolkit.utils import ModelNotFoundError, ToolkitError, get_output_filename
from video_toolkit.video_io import ProcessingResult, VideoReader, VideoWriter


class DepthV2Estimator:
    """Depth estimation using Depth Anything V2.

    10x faster than Stable Diffusion-based models with better accuracy.
    Trained on 595K synthetic + 62M real images.

    GitHub: https://github.com/DepthAnything/Depth-Anything-V2
    """

    MODEL_ID = "depth-v2"
    MODEL_NAME = "Depth Anything V2"

    def __init__(self, config: Optional[Config] = None):
        """Initialize Depth Anything V2 estimator.

        Args:
            config: Configuration object
        """
        self.config = config or Config.from_env()
        self._model = None

    def _load_model(self, variant: str = "large") -> None:
        """Load Depth Anything V2 model.

        Args:
            variant: Model variant (small, base, large)
        """
        if self._model is not None:
            return

        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError:
            raise ToolkitError(
                "Depth Anything V2 not installed. Run:\n"
                "  pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git\n"
                "Or use: ./scripts/install_depth.sh"
            )

        print(f"Loading {self.MODEL_NAME} ({variant}) on {self.config.device}...")

        try:
            import torch
            model_dir = self.config.get_model_dir(self.MODEL_ID)

            # Model configurations
            configs = {
                "small": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
                "base": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
                "large": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            }

            cfg = configs.get(variant, configs["large"])

            self._model = DepthAnythingV2(**cfg)
            self._model.load_state_dict(
                torch.load(model_dir / f"depth_anything_v2_{cfg['encoder']}.pth")
            )
            self._model.to(self.config.device)
            self._model.eval()

        except Exception as e:
            raise ModelNotFoundError(f"Failed to load {self.MODEL_NAME}: {e}")

        print(f"{self.MODEL_NAME} loaded successfully")

    def estimate_depth(
        self,
        video_path: str | Path,
        variant: str = "large",
        colormap: int = cv2.COLORMAP_INFERNO,
        output_dir: Optional[Path] = None,
    ) -> ProcessingResult:
        """Estimate depth for video frames.

        Args:
            video_path: Path to input video
            variant: Model variant (small, base, large)
            colormap: OpenCV colormap for visualization
            output_dir: Output directory

        Returns:
            ProcessingResult with path to depth video
        """
        video_path = Path(video_path)
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        self._load_model(variant)

        result = ProcessingResult(video_path, output_dir, "depth")

        with VideoReader(video_path, self.config.max_frames) as reader:
            info = reader.info
            print(f"Video: {info['width']}x{info['height']} @ {info['fps']:.1f}fps")

            output_path = get_output_filename(
                video_path, "depth", f"v2_{variant}", output_dir
            )

            with VideoWriter(
                output_path, info["width"], info["height"], info["fps"]
            ) as writer:
                print("Estimating depth...")

                for frame_idx, frame in tqdm(reader.frames(), desc="Processing"):
                    # Estimate depth
                    with self._no_grad():
                        depth = self._model.infer_image(frame)

                    # Write colorized depth
                    writer.write_depth(depth, colormap)

            result.add_output(
                output_path, self.MODEL_ID, writer.frame_count,
                variant=variant
            )
            print(f"Created: {output_path.name}")

        result.save_metadata()
        result.save_summary()

        return result

    def _no_grad(self):
        """Context manager for inference without gradients."""
        import torch
        return torch.no_grad()

    def get_info(self) -> dict:
        """Get model information."""
        import torch

        return {
            "model": self.MODEL_NAME,
            "model_id": self.MODEL_ID,
            "paper": "NeurIPS 2024",
            "speed": "10x faster than SD-based models",
            "variants": ["small", "base", "large"],
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": self.config.device,
        }
