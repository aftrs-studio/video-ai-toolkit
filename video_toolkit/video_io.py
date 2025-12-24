"""Video input/output handling for Video AI Toolkit."""

import json
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
from tqdm import tqdm

from video_toolkit.utils import VideoNotFoundError, get_video_info


class VideoReader:
    """Read video frames from a file."""

    def __init__(self, video_path: Path, max_frames: Optional[int] = None):
        """Initialize video reader.

        Args:
            video_path: Path to input video
            max_frames: Maximum frames to read (None = all)
        """
        self.video_path = Path(video_path)
        self.max_frames = max_frames
        self._cap: Optional[cv2.VideoCapture] = None
        self._info: Optional[dict] = None

    @property
    def info(self) -> dict:
        """Get video information."""
        if self._info is None:
            self._info = get_video_info(self.video_path)
        return self._info

    def __enter__(self) -> "VideoReader":
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise VideoNotFoundError(f"Cannot open video: {self.video_path}")
        return self

    def __exit__(self, *args) -> None:
        if self._cap:
            self._cap.release()

    def frames(self, show_progress: bool = True) -> Generator[tuple[int, np.ndarray], None, None]:
        """Iterate over video frames.

        Args:
            show_progress: Show progress bar

        Yields:
            Tuple of (frame_index, frame_array)
        """
        if self._cap is None:
            raise RuntimeError("VideoReader not opened. Use 'with' statement.")

        total = self.max_frames or self.info["frame_count"]
        pbar = tqdm(total=total, desc="Reading frames", disable=not show_progress)

        frame_idx = 0
        while True:
            if self.max_frames and frame_idx >= self.max_frames:
                break

            ret, frame = self._cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame_idx, frame_rgb

            frame_idx += 1
            pbar.update(1)

        pbar.close()

    def read_all(self, show_progress: bool = True) -> list[np.ndarray]:
        """Read all frames into memory.

        Args:
            show_progress: Show progress bar

        Returns:
            List of RGB frame arrays
        """
        frames = []
        for _, frame in self.frames(show_progress):
            frames.append(frame)
        return frames


class VideoWriter:
    """Write video frames to file."""

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        codec: str = "mp4v",
    ):
        """Initialize video writer.

        Args:
            output_path: Path for output video
            width: Frame width
            height: Frame height
            fps: Frames per second
            codec: Video codec (default: mp4v)
        """
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self._writer: Optional[cv2.VideoWriter] = None
        self.frame_count = 0

    def __enter__(self) -> "VideoWriter":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.width, self.height),
        )
        return self

    def __exit__(self, *args) -> None:
        if self._writer:
            self._writer.release()

    def write_frame(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        """Write a frame with optional mask overlay.

        Args:
            frame: RGB frame array
            mask: Binary mask array (same height/width as frame)
        """
        if self._writer is None:
            raise RuntimeError("VideoWriter not opened. Use 'with' statement.")

        if mask is not None:
            masked_frame = frame.copy()
            mask_3ch = np.stack([mask] * 3, axis=-1)
            masked_frame = np.where(mask_3ch, masked_frame, 0)
            frame = masked_frame

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._writer.write(frame_bgr)
        self.frame_count += 1

    def write_depth(self, depth: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> None:
        """Write a depth map frame with colormap visualization.

        Args:
            depth: Depth array (single channel)
            colormap: OpenCV colormap to use
        """
        if self._writer is None:
            raise RuntimeError("VideoWriter not opened. Use 'with' statement.")

        # Normalize depth to 0-255
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_norm, colormap)
        self._writer.write(depth_color)
        self.frame_count += 1


class ProcessingResult:
    """Container for processing results."""

    def __init__(self, video_path: Path, output_dir: Path, task: str):
        """Initialize processing result.

        Args:
            video_path: Original video path
            output_dir: Output directory
            task: Task name (segment, depth, matte, inpaint)
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.task = task
        self.outputs: list[dict] = []

    def add_output(
        self,
        output_path: Path,
        model: str,
        frame_count: int,
        **kwargs,
    ) -> None:
        """Add an output file to results.

        Args:
            output_path: Path to output video
            model: Model used
            frame_count: Number of frames
            **kwargs: Additional metadata
        """
        self.outputs.append({
            "output_path": str(output_path),
            "model": model,
            "frame_count": frame_count,
            **kwargs,
        })

    def save_metadata(self) -> Path:
        """Save processing metadata to JSON file.

        Returns:
            Path to metadata file
        """
        metadata = {
            "input_video": str(self.video_path),
            "task": self.task,
            "outputs": self.outputs,
            "total_outputs": len(self.outputs),
        }

        metadata_path = self.output_dir / f"{self.video_path.stem}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata_path

    def save_summary(self) -> Path:
        """Save human-readable summary.

        Returns:
            Path to summary file
        """
        lines = [
            f"Video AI Toolkit - {self.task.upper()} Results",
            "=" * 50,
            "",
            f"Input: {self.video_path}",
            f"Task: {self.task}",
            f"Total outputs: {len(self.outputs)}",
            "",
            "Outputs:",
        ]

        for out in self.outputs:
            lines.append(f"  - {out['model']}: {out['output_path']}")

        summary_path = self.output_dir / f"{self.video_path.stem}_summary.txt"
        with open(summary_path, "w") as f:
            f.write("\n".join(lines))

        return summary_path
