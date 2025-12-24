"""FILM frame interpolation wrapper."""

from pathlib import Path
from typing import Optional

import numpy as np

from video_toolkit.config import Config
from video_toolkit.video_io import VideoReader, VideoWriter, ProcessingResult
from video_toolkit.utils import ToolkitError, get_output_filename


class FILMInterpolator:
    """FILM (Frame Interpolation for Large Motion) interpolator.

    Google's frame interpolation model optimized for large motion.
    Better quality than RIFE for scenes with significant motion.

    Features:
        - Large motion handling
        - Multi-scale pyramid approach
        - High-quality temporal consistency
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model = None

    def _load_model(self, style: str = "film_net") -> None:
        """Load FILM model."""
        if self._model is not None:
            return

        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError:
            raise ToolkitError(
                "TensorFlow not installed. Run: pip install tensorflow tensorflow-hub"
            )

        model_path = self.config.model_path / "film" / style
        if model_path.exists():
            self._model = tf.saved_model.load(str(model_path))
        else:
            self._model = hub.load(
                "https://tfhub.dev/google/film/1"
            )

    def interpolate_video(
        self,
        video_path: str | Path,
        multiplier: int = 2,
        target_fps: Optional[float] = None,
    ) -> ProcessingResult:
        """Interpolate video frames using FILM.

        Args:
            video_path: Path to input video
            multiplier: Frame multiplier (2, 4, 8)
            target_fps: Target FPS (overrides multiplier if set)

        Returns:
            ProcessingResult with interpolated video path
        """
        import tensorflow as tf

        video_path = Path(video_path)
        self._load_model()

        reader = VideoReader(video_path)
        video_info = reader.get_info()

        source_fps = video_info["fps"]
        if target_fps:
            multiplier = int(target_fps / source_fps)
            multiplier = max(2, min(multiplier, 8))

        output_fps = source_fps * multiplier

        output_path = get_output_filename(
            video_path,
            "interpolate",
            f"{output_fps:.0f}fps_film",
            self.config.output_dir,
        )

        writer = VideoWriter(
            output_path,
            fps=output_fps,
            width=video_info["width"],
            height=video_info["height"],
        )

        result = ProcessingResult(source_video=video_path)

        try:
            prev_frame = None

            for frame in reader:
                if prev_frame is not None:
                    interpolated = self._interpolate_pair(
                        prev_frame, frame, multiplier
                    )
                    for interp_frame in interpolated[:-1]:
                        writer.write(interp_frame)

                writer.write(frame)
                prev_frame = frame

            result.add_output(
                output_path,
                "interpolated",
                {
                    "multiplier": multiplier,
                    "source_fps": source_fps,
                    "output_fps": output_fps,
                    "model": "film",
                },
            )

        finally:
            reader.close()
            writer.close()

        return result

    def _interpolate_pair(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        multiplier: int,
    ) -> list[np.ndarray]:
        """Interpolate between two frames using recursive bisection."""
        import tensorflow as tf

        results = self._recursive_interpolate(frame1, frame2, multiplier)
        return results

    def _recursive_interpolate(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        depth: int,
    ) -> list[np.ndarray]:
        """Recursively interpolate to achieve desired multiplier."""
        import tensorflow as tf

        if depth <= 1:
            return [frame1, frame2]

        img1 = tf.cast(frame1, tf.float32)[tf.newaxis, ...] / 255.0
        img2 = tf.cast(frame2, tf.float32)[tf.newaxis, ...] / 255.0

        mid = self._model({"time": tf.constant([0.5]), "x0": img1, "x1": img2})
        mid_frame = (mid["image"][0].numpy() * 255).astype(np.uint8)

        left = self._recursive_interpolate(frame1, mid_frame, depth // 2)
        right = self._recursive_interpolate(mid_frame, frame2, depth // 2)

        return left[:-1] + right
