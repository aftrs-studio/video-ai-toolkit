"""Video Depth Anything - Consistent depth estimation for long videos."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from video_toolkit.config import Config
from video_toolkit.utils import ModelNotFoundError, ToolkitError, get_output_filename
from video_toolkit.video_io import ProcessingResult, VideoReader, VideoWriter


class VideoDepthEstimator:
    """Depth estimation using Video Depth Anything.

    Provides consistent depth estimation for super-long videos with
    temporal coherence. CVPR 2025 Highlight.

    GitHub: https://github.com/DepthAnything/Video-Depth-Anything
    """

    MODEL_ID = "video-depth"
    MODEL_NAME = "Video Depth Anything"

    def __init__(self, config: Optional[Config] = None):
        """Initialize Video Depth Anything estimator.

        Args:
            config: Configuration object
        """
        self.config = config or Config.from_env()
        self._model = None

    def _load_model(self, variant: str = "large") -> None:
        """Load Video Depth Anything model.

        Args:
            variant: Model variant (small, base, large)
        """
        if self._model is not None:
            return

        try:
            from video_depth_anything import VideoDepthAnything
        except ImportError:
            raise ToolkitError(
                "Video Depth Anything not installed. Run:\n"
                "  pip install git+https://github.com/DepthAnything/Video-Depth-Anything.git\n"
                "Or use: ./scripts/install_depth.sh"
            )

        print(f"Loading {self.MODEL_NAME} ({variant}) on {self.config.device}...")

        try:
            model_dir = self.config.get_model_dir(self.MODEL_ID)
            self._model = VideoDepthAnything.from_pretrained(
                f"depth-anything/Video-Depth-Anything-{variant.capitalize()}",
                cache_dir=str(model_dir),
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
                video_path, "depth", variant, output_dir
            )

            with VideoWriter(
                output_path, info["width"], info["height"], info["fps"]
            ) as writer:
                print("Estimating depth...")

                for frame_idx, frame in tqdm(reader.frames(), desc="Processing"):
                    # Estimate depth
                    with self._no_grad():
                        depth = self._model.infer(frame)

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
            "paper": "CVPR 2025 Highlight",
            "variants": ["small (30fps)", "base", "large (best quality)"],
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": self.config.device,
        }
