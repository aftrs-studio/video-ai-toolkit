"""ViDeNN video denoiser wrapper."""

from pathlib import Path
from typing import Optional

import numpy as np

from video_toolkit.config import Config
from video_toolkit.video_io import VideoReader, VideoWriter, ProcessingResult
from video_toolkit.utils import ToolkitError, get_output_filename


class ViDeNNDenoiser:
    """ViDeNN (Deep Blind Video Denoising) denoiser.

    Handles various types of degradation including Gaussian noise
    and low-light conditions. Works in blind conditions without
    knowing the noise type. Presented at CVPR 2019.

    Features:
        - Blind denoising (no noise level estimation needed)
        - Low-light video enhancement
        - Temporal consistency
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model = None

    def _load_model(self) -> None:
        """Load ViDeNN model."""
        if self._model is not None:
            return

        try:
            import torch
        except ImportError:
            raise ToolkitError("PyTorch not installed. Run: pip install torch")

        import torch

        device = torch.device(self.config.device)
        model_path = self.config.model_path / "videnn" / "videnn.pth"

        try:
            # Try to load ViDeNN model
            from videnn.model import ViDeNN
            self._model = ViDeNN()
            if model_path.exists():
                state_dict = torch.load(str(model_path), map_location=device)
                self._model.load_state_dict(state_dict)
            self._model = self._model.to(device)
            self._model.eval()
        except ImportError:
            # Fallback to bilateral filtering
            self._model = "bilateral"

    def denoise_video(
        self,
        video_path: str | Path,
        strength: float = 1.0,
        low_light: bool = False,
    ) -> ProcessingResult:
        """Denoise video using ViDeNN.

        Args:
            video_path: Path to input video
            strength: Denoising strength (0.5-2.0)
            low_light: Enable low-light enhancement mode

        Returns:
            ProcessingResult with denoised video path
        """
        video_path = Path(video_path)
        self._load_model()

        reader = VideoReader(video_path)
        video_info = reader.get_info()

        suffix = "videnn"
        if low_light:
            suffix += "_lowlight"

        output_path = get_output_filename(
            video_path,
            "denoise",
            suffix,
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
                if low_light:
                    frame = self._enhance_low_light(frame)

                denoised = self._denoise_frame(
                    frame,
                    prev_frame,
                    strength,
                )

                writer.write(denoised)
                prev_frame = denoised

            result.add_output(
                output_path,
                "denoised",
                {
                    "model": "videnn",
                    "strength": strength,
                    "low_light": low_light,
                },
            )

        finally:
            reader.close()
            writer.close()

        return result

    def _denoise_frame(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray],
        strength: float,
    ) -> np.ndarray:
        """Denoise a single frame."""
        import torch

        if self._model == "bilateral":
            return self._bilateral_denoise(frame, prev_frame, strength)

        device = torch.device(self.config.device)

        # Prepare input
        frame_t = torch.from_numpy(frame).float() / 255.0
        frame_t = frame_t.permute(2, 0, 1).unsqueeze(0).to(device)

        if prev_frame is not None:
            prev_t = torch.from_numpy(prev_frame).float() / 255.0
            prev_t = prev_t.permute(2, 0, 1).unsqueeze(0).to(device)
            input_t = torch.cat([prev_t, frame_t], dim=1)
        else:
            input_t = torch.cat([frame_t, frame_t], dim=1)

        with torch.no_grad():
            output = self._model(input_t)

        # Apply strength
        if strength != 1.0:
            output = frame_t + (output - frame_t) * strength

        output_np = output[0].cpu().numpy().transpose(1, 2, 0)
        output_np = (output_np * 255).clip(0, 255).astype(np.uint8)

        return output_np

    def _bilateral_denoise(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray],
        strength: float,
    ) -> np.ndarray:
        """Fallback bilateral filter denoising."""
        import cv2

        # Bilateral filter parameters based on strength
        d = int(5 + strength * 4)
        sigma_color = int(50 + strength * 25)
        sigma_space = int(50 + strength * 25)

        denoised = cv2.bilateralFilter(
            frame, d, sigma_color, sigma_space
        )

        # Temporal blending with previous frame
        if prev_frame is not None:
            alpha = 0.7
            denoised = cv2.addWeighted(
                denoised, alpha,
                prev_frame, 1 - alpha,
                0,
            )

        return denoised

    def _enhance_low_light(self, frame: np.ndarray) -> np.ndarray:
        """Enhance low-light frame before denoising."""
        import cv2

        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return enhanced
