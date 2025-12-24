"""Video2X video upscaler wrapper."""

from pathlib import Path
from typing import Optional

from video_toolkit.config import Config
from video_toolkit.video_io import VideoReader, VideoWriter, ProcessingResult
from video_toolkit.utils import ToolkitError, get_output_filename


class Video2XUpscaler:
    """Video2X video upscaler.

    Multi-backend video upscaling framework supporting:
    - Waifu2x (anime-focused)
    - Real-ESRGAN (general)
    - SRMD (lightweight)

    This wrapper provides a simplified interface to Video2X's
    comprehensive upscaling pipeline.
    """

    BACKENDS = ["realesrgan", "waifu2x", "srmd"]

    def __init__(self, config: Config) -> None:
        self.config = config
        self._upscaler = None

    def _load_upscaler(self, backend: str = "realesrgan") -> None:
        """Load Video2X upscaler with specified backend."""
        if self._upscaler is not None:
            return

        try:
            from video2x import Video2X
        except ImportError:
            raise ToolkitError(
                "Video2X not installed. Run: pip install video2x"
            )

        self._upscaler = Video2X()
        self._backend = backend

    def upscale_video(
        self,
        video_path: str | Path,
        scale: int = 4,
        backend: str = "realesrgan",
        denoise: int = 0,
    ) -> ProcessingResult:
        """Upscale video using Video2X.

        Args:
            video_path: Path to input video
            scale: Upscaling factor (2, 3, or 4)
            backend: Upscaling backend (realesrgan, waifu2x, srmd)
            denoise: Denoise level (0-3, only for waifu2x)

        Returns:
            ProcessingResult with output video path
        """
        video_path = Path(video_path)
        self._load_upscaler(backend)

        if backend not in self.BACKENDS:
            raise ToolkitError(f"Unknown backend: {backend}. Use: {self.BACKENDS}")

        output_path = get_output_filename(
            video_path,
            "upscale",
            f"{scale}x_{backend}",
            self.config.output_dir,
        )

        result = ProcessingResult(source_video=video_path)

        try:
            self._upscaler.upscale(
                input_path=str(video_path),
                output_path=str(output_path),
                width=None,
                height=None,
                ratio=scale,
                processes=1,
                driver=backend,
                gpu=0 if self.config.device == "cuda" else -1,
                denoise_level=denoise if backend == "waifu2x" else None,
            )

            result.add_output(
                output_path,
                "upscaled",
                {"scale": scale, "backend": backend},
            )

        except Exception as e:
            raise ToolkitError(f"Video2X upscaling failed: {e}")

        return result
