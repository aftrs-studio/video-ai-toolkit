"""Real-ESRGAN video upscaler wrapper."""

from pathlib import Path
from typing import Optional

import numpy as np

from video_toolkit.config import Config
from video_toolkit.video_io import VideoReader, VideoWriter, ProcessingResult
from video_toolkit.utils import ToolkitError, get_output_filename


class RealESRGANUpscaler:
    """Real-ESRGAN video upscaler.

    Supports multiple models for general and anime content.
    Uses Real-ESRGAN for practical image/video restoration.

    Presets:
        - general: RealESRGAN_x4plus (default)
        - anime: RealESRGAN_x4plus_anime_6B
        - face: RealESRGAN_x4plus with GFPGAN face enhancement
    """

    MODELS = {
        "general": "RealESRGAN_x4plus",
        "anime": "RealESRGAN_x4plus_anime_6B",
        "fast": "realesr-animevideov3",
    }

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model = None
        self._face_enhancer = None

    def _load_model(self, preset: str = "general", scale: int = 4) -> None:
        """Load Real-ESRGAN model."""
        if self._model is not None:
            return

        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
        except ImportError:
            raise ToolkitError(
                "Real-ESRGAN not installed. Run: pip install realesrgan basicsr"
            )

        model_name = self.MODELS.get(preset, self.MODELS["general"])

        if preset == "anime":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=6,
                num_grow_ch=32,
                scale=4,
            )
        else:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )

        model_path = self.config.model_path / "realesrgan" / f"{model_name}.pth"

        self._model = RealESRGANer(
            scale=scale,
            model_path=str(model_path) if model_path.exists() else None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=self.config.device == "cuda",
            device=self.config.device,
        )

    def upscale_video(
        self,
        video_path: str | Path,
        scale: int = 4,
        preset: str = "general",
        enhance_face: bool = False,
    ) -> ProcessingResult:
        """Upscale video using Real-ESRGAN.

        Args:
            video_path: Path to input video
            scale: Upscaling factor (2 or 4)
            preset: Model preset (general, anime, fast)
            enhance_face: Whether to enhance faces with GFPGAN

        Returns:
            ProcessingResult with output video path
        """
        video_path = Path(video_path)
        self._load_model(preset, scale)

        reader = VideoReader(video_path)
        video_info = reader.get_info()

        output_path = get_output_filename(
            video_path,
            "upscale",
            f"{scale}x_{preset}",
            self.config.output_dir,
        )

        new_width = int(video_info["width"] * scale)
        new_height = int(video_info["height"] * scale)

        writer = VideoWriter(
            output_path,
            fps=video_info["fps"],
            width=new_width,
            height=new_height,
        )

        result = ProcessingResult(source_video=video_path)

        try:
            for frame in reader:
                upscaled, _ = self._model.enhance(frame, outscale=scale)

                if enhance_face and self._face_enhancer is not None:
                    _, _, upscaled = self._face_enhancer.enhance(
                        upscaled, has_aligned=False, paste_back=True
                    )

                writer.write(upscaled)

            result.add_output(output_path, "upscaled", {"scale": scale, "preset": preset})

        finally:
            reader.close()
            writer.close()

        return result
