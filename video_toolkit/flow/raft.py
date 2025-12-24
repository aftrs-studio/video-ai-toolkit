"""RAFT optical flow estimation wrapper."""

from pathlib import Path
from typing import Optional

import numpy as np

from video_toolkit.config import Config
from video_toolkit.video_io import VideoReader, VideoWriter, ProcessingResult
from video_toolkit.utils import ToolkitError, get_output_filename


class RAFTEstimator:
    """RAFT (Recurrent All-Pairs Field Transforms) optical flow estimator.

    ECCV 2020 Best Paper award winner for accurate optical flow.
    Provides dense motion estimation between consecutive frames.

    Features:
        - High-accuracy flow estimation
        - Recurrent refinement
        - Multiple model sizes (small, standard)
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model = None

    def _load_model(self, variant: str = "standard") -> None:
        """Load RAFT model."""
        if self._model is not None:
            return

        try:
            import torch
            from raft import RAFT
        except ImportError:
            raise ToolkitError(
                "RAFT not installed. Clone and install from: "
                "https://github.com/princeton-vl/RAFT"
            )

        import torch
        from argparse import Namespace

        device = torch.device(self.config.device)

        args = Namespace(
            model=str(self.config.model_path / "raft" / f"raft-{variant}.pth"),
            small=variant == "small",
            mixed_precision=True,
            alternate_corr=False,
        )

        self._model = RAFT(args)

        model_path = self.config.model_path / "raft" / f"raft-{variant}.pth"
        if model_path.exists():
            self._model.load_state_dict(torch.load(str(model_path), map_location=device))

        self._model = self._model.to(device)
        self._model.eval()

    def estimate_flow(
        self,
        video_path: str | Path,
        variant: str = "standard",
        visualize: bool = True,
    ) -> ProcessingResult:
        """Estimate optical flow for video using RAFT.

        Args:
            video_path: Path to input video
            variant: Model variant (small, standard)
            visualize: Whether to output flow visualization

        Returns:
            ProcessingResult with flow video/data
        """
        import torch

        video_path = Path(video_path)
        self._load_model(variant)

        reader = VideoReader(video_path)
        video_info = reader.get_info()

        output_path = get_output_filename(
            video_path,
            "flow",
            f"raft_{variant}",
            self.config.output_dir,
        )

        writer = VideoWriter(
            output_path,
            fps=video_info["fps"],
            width=video_info["width"],
            height=video_info["height"],
        )

        result = ProcessingResult(source_video=video_path)

        try:
            prev_frame = None

            for frame in reader:
                if prev_frame is not None:
                    flow = self._compute_flow(prev_frame, frame)

                    if visualize:
                        flow_vis = self._visualize_flow(flow)
                        writer.write(flow_vis)
                else:
                    black = np.zeros_like(frame)
                    writer.write(black)

                prev_frame = frame

            result.add_output(
                output_path,
                "optical_flow",
                {"model": "raft", "variant": variant},
            )

        finally:
            reader.close()
            writer.close()

        return result

    def _compute_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> np.ndarray:
        """Compute optical flow between two frames."""
        import torch

        device = torch.device(self.config.device)

        img1 = torch.from_numpy(frame1.transpose(2, 0, 1)).float().unsqueeze(0)
        img2 = torch.from_numpy(frame2.transpose(2, 0, 1)).float().unsqueeze(0)

        img1 = img1.to(device)
        img2 = img2.to(device)

        with torch.no_grad():
            _, flow = self._model(img1, img2, iters=20, test_mode=True)

        flow_np = flow[0].cpu().numpy().transpose(1, 2, 0)
        return flow_np

    def _visualize_flow(self, flow: np.ndarray) -> np.ndarray:
        """Visualize optical flow using HSV color space."""
        import cv2

        h, w = flow.shape[:2]
        fx, fy = flow[:, :, 0], flow[:, :, 1]

        mag, ang = cv2.cartToPolar(fx, fy)
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return rgb
