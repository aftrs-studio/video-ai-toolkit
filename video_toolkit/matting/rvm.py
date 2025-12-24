"""RobustVideoMatting - Real-time video matting."""

from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from video_toolkit.config import Config
from video_toolkit.utils import ModelNotFoundError, ToolkitError, get_output_filename
from video_toolkit.video_io import ProcessingResult, VideoReader, VideoWriter


class RVMatter:
    """Video matting using RobustVideoMatting.

    Real-time, high-resolution human video matting with temporal guidance.
    Achieves 4K@76fps and HD@104fps on GTX 1080Ti.

    GitHub: https://github.com/PeterL1n/RobustVideoMatting
    """

    MODEL_ID = "rvm"
    MODEL_NAME = "RobustVideoMatting"

    def __init__(self, config: Optional[Config] = None):
        """Initialize RobustVideoMatting.

        Args:
            config: Configuration object
        """
        self.config = config or Config.from_env()
        self._model = None

    def _load_model(self, backbone: str = "mobilenetv3") -> None:
        """Load RVM model.

        Args:
            backbone: Model backbone (mobilenetv3 or resnet50)
        """
        if self._model is not None:
            return

        try:
            import torch
        except ImportError:
            raise ToolkitError("PyTorch not installed")

        print(f"Loading {self.MODEL_NAME} ({backbone}) on {self.config.device}...")

        try:
            model_dir = self.config.get_model_dir(self.MODEL_ID)

            # Load model via torch.hub
            self._model = torch.hub.load(
                "PeterL1n/RobustVideoMatting",
                "mobilenetv3" if backbone == "mobilenetv3" else "resnet50",
            )
            self._model.to(self.config.device)
            self._model.eval()

        except Exception as e:
            raise ModelNotFoundError(f"Failed to load {self.MODEL_NAME}: {e}")

        print(f"{self.MODEL_NAME} loaded successfully")

    def remove_background(
        self,
        video_path: str | Path,
        backbone: str = "mobilenetv3",
        background_color: tuple[int, int, int] = (0, 255, 0),
        output_dir: Optional[Path] = None,
    ) -> ProcessingResult:
        """Remove background from video.

        Args:
            video_path: Path to input video
            backbone: Model backbone (mobilenetv3 or resnet50)
            background_color: RGB color for background (default: green)
            output_dir: Output directory

        Returns:
            ProcessingResult with path to matted video
        """
        video_path = Path(video_path)
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        self._load_model(backbone)

        result = ProcessingResult(video_path, output_dir, "matte")

        with VideoReader(video_path, self.config.max_frames) as reader:
            info = reader.info
            print(f"Video: {info['width']}x{info['height']} @ {info['fps']:.1f}fps")

            output_path = get_output_filename(
                video_path, "matte", backbone, output_dir
            )

            with VideoWriter(
                output_path, info["width"], info["height"], info["fps"]
            ) as writer:
                print("Removing background...")

                # Initialize recurrent states
                rec = [None] * 4

                for frame_idx, frame in tqdm(reader.frames(), desc="Processing"):
                    with self._no_grad():
                        # Convert to tensor
                        import torch
                        src = torch.from_numpy(frame).permute(2, 0, 1).float() / 255
                        src = src.unsqueeze(0).to(self.config.device)

                        # Run model
                        fgr, pha, *rec = self._model(src, *rec, downsample_ratio=0.25)

                        # Composite with background color
                        bg = torch.tensor(background_color, device=self.config.device).float() / 255
                        bg = bg.view(1, 3, 1, 1).expand_as(fgr)
                        com = fgr * pha + bg * (1 - pha)

                        # Convert back to numpy
                        com = (com[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                    writer.write_frame(com)

            result.add_output(
                output_path, self.MODEL_ID, writer.frame_count,
                backbone=backbone
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
            "paper": "WACV 2022",
            "speed": "4K@76fps, HD@104fps (GTX 1080Ti)",
            "backbones": ["mobilenetv3 (faster)", "resnet50 (better quality)"],
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": self.config.device,
        }
