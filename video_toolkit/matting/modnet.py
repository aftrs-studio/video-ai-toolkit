"""MODNet - Portrait matting."""

from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from video_toolkit.config import Config
from video_toolkit.utils import ModelNotFoundError, ToolkitError, get_output_filename
from video_toolkit.video_io import ProcessingResult, VideoReader, VideoWriter


class MODNetMatter:
    """Video matting using MODNet.

    Real-time trimap-free portrait matting via objective decomposition.
    Optimized for portrait/human subjects.

    GitHub: https://github.com/ZHKKKe/MODNet
    """

    MODEL_ID = "modnet"
    MODEL_NAME = "MODNet"

    def __init__(self, config: Optional[Config] = None):
        """Initialize MODNet.

        Args:
            config: Configuration object
        """
        self.config = config or Config.from_env()
        self._model = None

    def _load_model(self) -> None:
        """Load MODNet model."""
        if self._model is not None:
            return

        try:
            import torch
            from modnet.models.modnet import MODNet
        except ImportError:
            raise ToolkitError(
                "MODNet not installed. Run:\n"
                "  pip install git+https://github.com/ZHKKKe/MODNet.git\n"
                "Or use: ./scripts/install_matting.sh"
            )

        print(f"Loading {self.MODEL_NAME} on {self.config.device}...")

        try:
            model_dir = self.config.get_model_dir(self.MODEL_ID)

            self._model = MODNet(backbone_pretrained=False)
            self._model.load_state_dict(
                torch.load(model_dir / "modnet_photographic_portrait_matting.ckpt")
            )
            self._model.to(self.config.device)
            self._model.eval()

        except Exception as e:
            raise ModelNotFoundError(f"Failed to load {self.MODEL_NAME}: {e}")

        print(f"{self.MODEL_NAME} loaded successfully")

    def remove_background(
        self,
        video_path: str | Path,
        background_color: tuple[int, int, int] = (0, 255, 0),
        output_dir: Optional[Path] = None,
    ) -> ProcessingResult:
        """Remove background from portrait video.

        Args:
            video_path: Path to input video
            background_color: RGB color for background (default: green)
            output_dir: Output directory

        Returns:
            ProcessingResult with path to matted video
        """
        video_path = Path(video_path)
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        self._load_model()

        result = ProcessingResult(video_path, output_dir, "matte")

        with VideoReader(video_path, self.config.max_frames) as reader:
            info = reader.info
            print(f"Video: {info['width']}x{info['height']} @ {info['fps']:.1f}fps")

            output_path = get_output_filename(
                video_path, "matte", "modnet", output_dir
            )

            with VideoWriter(
                output_path, info["width"], info["height"], info["fps"]
            ) as writer:
                print("Removing background...")

                for frame_idx, frame in tqdm(reader.frames(), desc="Processing"):
                    with self._no_grad():
                        import torch
                        import torch.nn.functional as F

                        # Preprocess
                        im = torch.from_numpy(frame).permute(2, 0, 1).float() / 255
                        im = im.unsqueeze(0).to(self.config.device)

                        # Normalize
                        im = (im - 0.5) / 0.5

                        # Resize for inference
                        _, _, h, w = im.shape
                        if max(h, w) > 512:
                            scale = 512 / max(h, w)
                            im_small = F.interpolate(im, scale_factor=scale, mode="area")
                        else:
                            im_small = im

                        # Run model
                        _, _, matte = self._model(im_small, True)

                        # Resize matte back
                        matte = F.interpolate(matte, size=(h, w), mode="area")

                        # Composite with background
                        bg = torch.tensor(background_color, device=self.config.device).float() / 255
                        bg = bg.view(1, 3, 1, 1).expand(1, 3, h, w)

                        fg = (im * 0.5 + 0.5)  # Denormalize
                        com = fg * matte + bg * (1 - matte)

                        # Convert back to numpy
                        com = (com[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                    writer.write_frame(com)

            result.add_output(output_path, self.MODEL_ID, writer.frame_count)
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
            "paper": "AAAI 2022",
            "best_for": "Portrait/human subjects",
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": self.config.device,
        }
