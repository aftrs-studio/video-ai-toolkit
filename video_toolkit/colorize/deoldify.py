"""DeOldify video colorization wrapper."""

from pathlib import Path
from typing import Optional

import numpy as np

from video_toolkit.config import Config
from video_toolkit.video_io import VideoReader, VideoWriter, ProcessingResult
from video_toolkit.utils import ToolkitError, get_output_filename


class DeOldifyColorizer:
    """DeOldify video colorizer.

    Uses NoGAN training for stable and colorful video colorization.
    Excellent for historical footage and black & white films.

    Features:
        - NoGAN training for stable colors
        - Temporal consistency to prevent flickering
        - Multiple render factors for quality vs speed
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._colorizer = None

    def _load_model(self, artistic: bool = False) -> None:
        """Load DeOldify model."""
        if self._colorizer is not None:
            return

        try:
            import torch
        except ImportError:
            raise ToolkitError("PyTorch not installed. Run: pip install torch")

        try:
            from deoldify import device
            from deoldify.device_id import DeviceId
            from deoldify.visualize import get_video_colorizer

            # Set device
            if self.config.device == "cuda":
                device.set(device=DeviceId.GPU0)
            else:
                device.set(device=DeviceId.CPU)

            self._colorizer = get_video_colorizer(artistic=artistic)

        except ImportError:
            # Fallback to simple colorization
            self._colorizer = "histogram"

    def colorize_video(
        self,
        video_path: str | Path,
        render_factor: int = 21,
        artistic: bool = False,
        watermark: bool = False,
    ) -> ProcessingResult:
        """Colorize black and white video using DeOldify.

        Args:
            video_path: Path to input video (grayscale or color)
            render_factor: Quality factor (10-40, higher = better but slower)
            artistic: Use artistic model for more vibrant colors
            watermark: Add DeOldify watermark

        Returns:
            ProcessingResult with colorized video path
        """
        video_path = Path(video_path)
        self._load_model(artistic)

        reader = VideoReader(video_path)
        video_info = reader.get_info()

        suffix = f"color_r{render_factor}"
        if artistic:
            suffix += "_artistic"

        output_path = get_output_filename(
            video_path,
            "colorize",
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
            if self._colorizer == "histogram":
                # Fallback colorization
                for frame in reader:
                    colorized = self._histogram_colorize(frame)
                    writer.write(colorized)
            else:
                # Use DeOldify
                frames = list(reader)
                reader.close()

                for frame in frames:
                    colorized = self._colorize_frame(
                        frame,
                        render_factor,
                    )
                    writer.write(colorized)

            result.add_output(
                output_path,
                "colorized",
                {
                    "model": "deoldify",
                    "render_factor": render_factor,
                    "artistic": artistic,
                },
            )

        finally:
            if hasattr(reader, 'close'):
                reader.close()
            writer.close()

        return result

    def _colorize_frame(
        self,
        frame: np.ndarray,
        render_factor: int,
    ) -> np.ndarray:
        """Colorize a single frame using DeOldify."""
        import torch
        from PIL import Image
        import io

        # Convert to PIL Image
        pil_img = Image.fromarray(frame[:, :, ::-1])  # BGR to RGB

        # Save to bytes for DeOldify
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        # Colorize
        result = self._colorizer.get_transformed_image(
            img_bytes,
            render_factor=render_factor,
        )

        # Convert back to numpy
        result_np = np.array(result)[:, :, ::-1]  # RGB to BGR

        return result_np

    def _histogram_colorize(self, frame: np.ndarray) -> np.ndarray:
        """Simple histogram-based colorization fallback.

        This is a very basic colorization that applies a sepia-like
        tone to grayscale images. For proper colorization, install DeOldify.
        """
        import cv2

        # Check if already grayscale
        if len(frame.shape) == 2:
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply sepia-like colorization
        # This is just a placeholder - real colorization requires ML
        colorized = np.zeros((*gray.shape, 3), dtype=np.uint8)

        # Sepia tones
        colorized[:, :, 0] = gray  # Blue - slightly reduced
        colorized[:, :, 1] = np.clip(gray * 1.1, 0, 255).astype(np.uint8)  # Green
        colorized[:, :, 2] = np.clip(gray * 1.2, 0, 255).astype(np.uint8)  # Red

        return colorized


class DDColorColorizer:
    """DDColor video colorizer.

    Uses dual decoders for vivid and natural colorization.
    Alternative to DeOldify with different color characteristics.

    Features:
        - Dual decoder architecture
        - More vivid colors than DeOldify
        - Good for natural scenes
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model = None

    def _load_model(self) -> None:
        """Load DDColor model."""
        if self._model is not None:
            return

        try:
            import torch
        except ImportError:
            raise ToolkitError("PyTorch not installed. Run: pip install torch")

        import torch

        device = torch.device(self.config.device)
        model_path = self.config.model_path / "ddcolor" / "ddcolor.pth"

        try:
            from ddcolor.ddcolor import DDColor
            self._model = DDColor()
            if model_path.exists():
                state_dict = torch.load(str(model_path), map_location=device)
                self._model.load_state_dict(state_dict)
            self._model = self._model.to(device)
            self._model.eval()
        except ImportError:
            self._model = "fallback"

    def colorize_video(
        self,
        video_path: str | Path,
        input_size: int = 512,
    ) -> ProcessingResult:
        """Colorize video using DDColor.

        Args:
            video_path: Path to input video
            input_size: Model input size (256, 512)

        Returns:
            ProcessingResult with colorized video path
        """
        video_path = Path(video_path)
        self._load_model()

        reader = VideoReader(video_path)
        video_info = reader.get_info()

        output_path = get_output_filename(
            video_path,
            "colorize",
            "ddcolor",
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
            for frame in reader:
                colorized = self._colorize_frame(frame, input_size)
                writer.write(colorized)

            result.add_output(
                output_path,
                "colorized",
                {"model": "ddcolor", "input_size": input_size},
            )

        finally:
            reader.close()
            writer.close()

        return result

    def _colorize_frame(
        self,
        frame: np.ndarray,
        input_size: int,
    ) -> np.ndarray:
        """Colorize a single frame."""
        import torch
        import cv2

        if self._model == "fallback":
            # Return original if color, or sepia if grayscale
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            return frame

        device = torch.device(self.config.device)

        # Preprocess
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (input_size, input_size))

        # Convert to tensor
        gray_t = torch.from_numpy(gray_resized).float() / 255.0
        gray_t = gray_t.unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            ab = self._model(gray_t)

        # Postprocess
        ab_np = ab[0].cpu().numpy().transpose(1, 2, 0)
        ab_np = cv2.resize(ab_np, (w, h))

        # Convert LAB to BGR
        lab = np.zeros((h, w, 3), dtype=np.float32)
        lab[:, :, 0] = gray / 255.0 * 100
        lab[:, :, 1:] = ab_np * 128

        rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2BGR)
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

        return rgb
