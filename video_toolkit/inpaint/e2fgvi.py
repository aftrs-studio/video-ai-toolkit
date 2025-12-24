"""E2FGVI - End-to-end flow-guided video inpainting."""

from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from video_toolkit.config import Config
from video_toolkit.utils import ModelNotFoundError, ToolkitError, get_output_filename
from video_toolkit.video_io import ProcessingResult, VideoReader, VideoWriter


class E2FGVIInpainter:
    """Video inpainting using E2FGVI.

    End-to-end framework for flow-guided video inpainting with
    temporal consistency through pixel propagation.

    GitHub: https://github.com/MCG-NKU/E2FGVI
    """

    MODEL_ID = "e2fgvi"
    MODEL_NAME = "E2FGVI"

    def __init__(self, config: Optional[Config] = None):
        """Initialize E2FGVI.

        Args:
            config: Configuration object
        """
        self.config = config or Config.from_env()
        self._model = None

    def _load_model(self) -> None:
        """Load E2FGVI model."""
        if self._model is not None:
            return

        try:
            import torch
            from e2fgvi.model import E2FGVI
        except ImportError:
            raise ToolkitError(
                "E2FGVI not installed. Run:\n"
                "  pip install git+https://github.com/MCG-NKU/E2FGVI.git\n"
                "Or use: ./scripts/install_inpaint.sh"
            )

        print(f"Loading {self.MODEL_NAME} on {self.config.device}...")

        try:
            model_dir = self.config.get_model_dir(self.MODEL_ID)

            self._model = E2FGVI()
            self._model.load_state_dict(
                torch.load(model_dir / "E2FGVI-HQ-CVPR22.pth")
            )
            self._model.to(self.config.device)
            self._model.eval()

        except Exception as e:
            raise ModelNotFoundError(f"Failed to load {self.MODEL_NAME}: {e}")

        print(f"{self.MODEL_NAME} loaded successfully")

    def inpaint_video(
        self,
        video_path: str | Path,
        mask_path: str | Path,
        output_dir: Optional[Path] = None,
    ) -> ProcessingResult:
        """Inpaint (remove objects from) video using mask.

        Args:
            video_path: Path to input video
            mask_path: Path to mask video (white = area to inpaint)
            output_dir: Output directory

        Returns:
            ProcessingResult with path to inpainted video
        """
        video_path = Path(video_path)
        mask_path = Path(mask_path)
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        self._load_model()

        result = ProcessingResult(video_path, output_dir, "inpaint")

        # Read video and mask
        with VideoReader(video_path, self.config.max_frames) as reader:
            info = reader.info
            frames = reader.read_all()

        with VideoReader(mask_path, self.config.max_frames) as mask_reader:
            masks = mask_reader.read_all()

        print(f"Video: {info['width']}x{info['height']} @ {info['fps']:.1f}fps")
        print(f"Loaded {len(frames)} frames and {len(masks)} masks")

        # Ensure same number of frames
        min_frames = min(len(frames), len(masks))
        frames = frames[:min_frames]
        masks = masks[:min_frames]

        output_path = get_output_filename(
            video_path, "inpaint", "e2fgvi", output_dir
        )

        with VideoWriter(
            output_path, info["width"], info["height"], info["fps"]
        ) as writer:
            print("Inpainting video...")

            with self._no_grad():
                import torch

                # Process in batches
                batch_size = 5
                completed_frames = []

                for i in tqdm(range(0, len(frames), batch_size), desc="Processing"):
                    batch_frames = frames[i:i+batch_size]
                    batch_masks = masks[i:i+batch_size]

                    # Convert to tensors
                    frames_t = torch.stack([
                        torch.from_numpy(f).permute(2, 0, 1).float() / 255
                        for f in batch_frames
                    ]).to(self.config.device)

                    masks_t = torch.stack([
                        torch.from_numpy(m[:, :, 0]).float() / 255
                        for m in batch_masks
                    ]).unsqueeze(1).to(self.config.device)

                    # Run model
                    output = self._model(frames_t, masks_t)

                    # Convert back to numpy
                    for out in output:
                        frame = (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        completed_frames.append(frame)

                # Write output
                for frame in completed_frames:
                    writer.write_frame(frame)

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
            "paper": "CVPR 2022",
            "features": "Flow-guided temporal consistency",
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": self.config.device,
        }
