"""GFPGAN face restoration wrapper."""

from pathlib import Path
from typing import Optional

import numpy as np

from video_toolkit.config import Config
from video_toolkit.video_io import VideoReader, VideoWriter, ProcessingResult
from video_toolkit.utils import ToolkitError, get_output_filename


class GFPGANRestorer:
    """GFPGAN face restorer.

    GAN-based face restoration with Real-ESRGAN background enhancement.
    Excellent for restoring low-quality or degraded faces in video.

    Features:
        - Blind face restoration
        - Background enhancement with Real-ESRGAN
        - Face detection and alignment
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._restorer = None

    def _load_model(self, version: str = "1.4") -> None:
        """Load GFPGAN model."""
        if self._restorer is not None:
            return

        try:
            from gfpgan import GFPGANer
        except ImportError:
            raise ToolkitError("GFPGAN not installed. Run: pip install gfpgan")

        model_path = self.config.model_path / "gfpgan" / f"GFPGANv{version}.pth"

        self._restorer = GFPGANer(
            model_path=str(model_path) if model_path.exists() else None,
            upscale=2,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=self._get_bg_upsampler(),
            device=self.config.device,
        )

    def _get_bg_upsampler(self):
        """Get Real-ESRGAN background upsampler."""
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )

            return RealESRGANer(
                scale=2,
                model_path=None,
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=self.config.device == "cuda",
                device=self.config.device,
            )
        except ImportError:
            return None

    def restore_video(
        self,
        video_path: str | Path,
        upscale: int = 2,
        enhance_background: bool = True,
    ) -> ProcessingResult:
        """Restore faces in video using GFPGAN.

        Args:
            video_path: Path to input video
            upscale: Output upscale factor (1, 2, or 4)
            enhance_background: Whether to enhance background with Real-ESRGAN

        Returns:
            ProcessingResult with restored video path
        """
        video_path = Path(video_path)
        self._load_model()

        reader = VideoReader(video_path)
        video_info = reader.get_info()

        output_path = get_output_filename(
            video_path,
            "face",
            f"gfpgan_{upscale}x",
            self.config.output_dir,
        )

        new_width = int(video_info["width"] * upscale)
        new_height = int(video_info["height"] * upscale)

        writer = VideoWriter(
            output_path,
            fps=video_info["fps"],
            width=new_width,
            height=new_height,
        )

        result = ProcessingResult(source_video=video_path)
        faces_restored = 0

        try:
            for frame in reader:
                _, _, restored = self._restorer.enhance(
                    frame,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                )

                if restored is not None:
                    faces_restored += 1
                    writer.write(restored)
                else:
                    import cv2
                    resized = cv2.resize(frame, (new_width, new_height))
                    writer.write(resized)

            result.add_output(
                output_path,
                "face_restored",
                {
                    "model": "gfpgan",
                    "upscale": upscale,
                    "frames_with_faces": faces_restored,
                },
            )

        finally:
            reader.close()
            writer.close()

        return result
