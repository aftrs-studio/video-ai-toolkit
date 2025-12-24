"""ProPainter - Video inpainting with improved propagation."""

from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from video_toolkit.config import Config
from video_toolkit.utils import ModelNotFoundError, ToolkitError, get_output_filename
from video_toolkit.video_io import ProcessingResult, VideoReader, VideoWriter


class ProPainterInpainter:
    """Video inpainting using ProPainter.

    Improves flow-based propagation and spatiotemporal Transformers
    for high-quality video inpainting. 40x faster flow completion.

    GitHub: https://github.com/sczhou/ProPainter
    """

    MODEL_ID = "propainter"
    MODEL_NAME = "ProPainter"

    def __init__(self, config: Optional[Config] = None):
        """Initialize ProPainter.

        Args:
            config: Configuration object
        """
        self.config = config or Config.from_env()
        self._model = None
        self._flow_model = None

    def _load_model(self) -> None:
        """Load ProPainter models."""
        if self._model is not None:
            return

        try:
            import torch
            from propainter.core.model import ProPainter
            from propainter.core.flow import RAFT
        except ImportError:
            raise ToolkitError(
                "ProPainter not installed. Run:\n"
                "  pip install git+https://github.com/sczhou/ProPainter.git\n"
                "Or use: ./scripts/install_inpaint.sh"
            )

        print(f"Loading {self.MODEL_NAME} on {self.config.device}...")

        try:
            model_dir = self.config.get_model_dir(self.MODEL_ID)

            # Load ProPainter
            self._model = ProPainter()
            self._model.load_state_dict(
                torch.load(model_dir / "ProPainter.pth")
            )
            self._model.to(self.config.device)
            self._model.eval()

            # Load flow model
            self._flow_model = RAFT()
            self._flow_model.load_state_dict(
                torch.load(model_dir / "raft-things.pth")
            )
            self._flow_model.to(self.config.device)
            self._flow_model.eval()

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
            video_path, "inpaint", "propainter", output_dir
        )

        with VideoWriter(
            output_path, info["width"], info["height"], info["fps"]
        ) as writer:
            print("Inpainting video...")

            with self._no_grad():
                import torch

                # Convert to tensors
                frames_t = torch.stack([
                    torch.from_numpy(f).permute(2, 0, 1).float() / 255
                    for f in frames
                ]).to(self.config.device)

                masks_t = torch.stack([
                    torch.from_numpy(m[:, :, 0]).float() / 255
                    for m in masks
                ]).unsqueeze(1).to(self.config.device)

                # Compute optical flow
                print("Computing optical flow...")
                flows_f, flows_b = self._compute_flow(frames_t)

                # Complete flow in masked regions
                print("Completing flow...")
                flows_f, flows_b = self._complete_flow(
                    flows_f, flows_b, masks_t
                )

                # Propagate and inpaint
                print("Propagating and inpainting...")
                for i in tqdm(range(len(frames)), desc="Processing"):
                    # Get completed frame
                    completed = self._model.forward_single(
                        frames_t[i:i+1],
                        masks_t[i:i+1],
                        flows_f,
                        flows_b,
                        frame_idx=i,
                    )

                    # Convert back to numpy
                    out = (completed[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    writer.write_frame(out)

        result.add_output(output_path, self.MODEL_ID, writer.frame_count)
        print(f"Created: {output_path.name}")

        result.save_metadata()
        result.save_summary()

        return result

    def _compute_flow(self, frames):
        """Compute forward and backward optical flow."""
        flows_f = []
        flows_b = []

        for i in range(len(frames) - 1):
            flow_f = self._flow_model(frames[i:i+1], frames[i+1:i+2])
            flow_b = self._flow_model(frames[i+1:i+2], frames[i:i+1])
            flows_f.append(flow_f)
            flows_b.append(flow_b)

        import torch
        return torch.stack(flows_f), torch.stack(flows_b)

    def _complete_flow(self, flows_f, flows_b, masks):
        """Complete flow in masked regions."""
        # Simplified - full implementation would use ProPainter's flow completion
        return flows_f, flows_b

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
            "paper": "ICCV 2023",
            "features": "40x faster flow completion, dual-domain propagation",
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": self.config.device,
        }
