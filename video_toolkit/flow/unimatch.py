"""UniMatch optical flow estimation wrapper."""

from pathlib import Path
from typing import Optional

import numpy as np

from video_toolkit.config import Config
from video_toolkit.video_io import VideoReader, VideoWriter, ProcessingResult
from video_toolkit.utils import ToolkitError, get_output_filename


class UniMatchEstimator:
    """UniMatch unified matching estimator.

    Unified model for optical flow, stereo depth, and feature matching.
    Provides consistent results across multiple vision tasks.

    Features:
        - Unified architecture for flow/depth/stereo
        - Cross-scale matching
        - Self-supervised learning support
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model = None

    def _load_model(self, task: str = "flow") -> None:
        """Load UniMatch model."""
        if self._model is not None:
            return

        try:
            import torch
            from unimatch.unimatch import UniMatch
        except ImportError:
            raise ToolkitError(
                "UniMatch not installed. Clone and install from: "
                "https://github.com/autonomousvision/unimatch"
            )

        import torch

        device = torch.device(self.config.device)

        self._model = UniMatch(
            num_scales=2,
            feature_channels=128,
            upsample_factor=4,
            num_head=1,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            reg_refine=True,
            task=task,
        )

        model_path = self.config.model_path / "unimatch" / f"unimatch-{task}.pth"
        if model_path.exists():
            checkpoint = torch.load(str(model_path), map_location=device)
            self._model.load_state_dict(checkpoint["model"])

        self._model = self._model.to(device)
        self._model.eval()

    def estimate_flow(
        self,
        video_path: str | Path,
        visualize: bool = True,
    ) -> ProcessingResult:
        """Estimate optical flow for video using UniMatch.

        Args:
            video_path: Path to input video
            visualize: Whether to output flow visualization

        Returns:
            ProcessingResult with flow video/data
        """
        import torch

        video_path = Path(video_path)
        self._load_model("flow")

        reader = VideoReader(video_path)
        video_info = reader.get_info()

        output_path = get_output_filename(
            video_path,
            "flow",
            "unimatch",
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
                {"model": "unimatch"},
            )

        finally:
            reader.close()
            writer.close()

        return result

    def estimate_depth(
        self,
        video_path: str | Path,
        visualize: bool = True,
    ) -> ProcessingResult:
        """Estimate stereo depth for video using UniMatch.

        Note: Requires stereo video input (side-by-side).

        Args:
            video_path: Path to stereo video
            visualize: Whether to output depth visualization

        Returns:
            ProcessingResult with depth video/data
        """
        video_path = Path(video_path)
        self._load_model("stereo")

        reader = VideoReader(video_path)
        video_info = reader.get_info()

        half_width = video_info["width"] // 2

        output_path = get_output_filename(
            video_path,
            "depth",
            "unimatch_stereo",
            self.config.output_dir,
        )

        writer = VideoWriter(
            output_path,
            fps=video_info["fps"],
            width=half_width,
            height=video_info["height"],
        )

        result = ProcessingResult(source_video=video_path)

        try:
            for frame in reader:
                left = frame[:, :half_width, :]
                right = frame[:, half_width:, :]

                disparity = self._compute_stereo(left, right)

                if visualize:
                    depth_vis = self._visualize_depth(disparity)
                    writer.write(depth_vis)

            result.add_output(
                output_path,
                "stereo_depth",
                {"model": "unimatch"},
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
            result = self._model(
                img1, img2,
                attn_type="swin",
                attn_splits_list=[2, 8],
                corr_radius_list=[-1, 4],
                prop_radius_list=[-1, 1],
            )

        flow = result["flow_preds"][-1]
        flow_np = flow[0].cpu().numpy().transpose(1, 2, 0)
        return flow_np

    def _compute_stereo(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        """Compute stereo disparity."""
        import torch

        device = torch.device(self.config.device)

        left_t = torch.from_numpy(left.transpose(2, 0, 1)).float().unsqueeze(0)
        right_t = torch.from_numpy(right.transpose(2, 0, 1)).float().unsqueeze(0)

        left_t = left_t.to(device)
        right_t = right_t.to(device)

        with torch.no_grad():
            result = self._model(
                left_t, right_t,
                attn_type="swin",
                attn_splits_list=[2, 8],
                corr_radius_list=[-1, 4],
                prop_radius_list=[-1, 1],
            )

        disparity = result["flow_preds"][-1]
        disp_np = disparity[0, 0].cpu().numpy()
        return disp_np

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

    def _visualize_depth(self, disparity: np.ndarray) -> np.ndarray:
        """Visualize disparity/depth map."""
        import cv2

        normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_MAGMA)
        return colored
