"""Utility functions for Video AI Toolkit."""

import glob as glob_module
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from video_toolkit.video_io import ProcessingResult


class ToolkitError(Exception):
    """Base exception for Video AI Toolkit."""
    pass


class VideoNotFoundError(ToolkitError):
    """Raised when input video file is not found."""
    pass


class ModelNotFoundError(ToolkitError):
    """Raised when model checkpoint is not found."""
    pass


class CUDANotAvailableError(ToolkitError):
    """Raised when CUDA is required but not available."""
    pass


def parse_concepts(concepts_str: Optional[str]) -> list[str]:
    """Parse comma-separated concepts string into list.

    Args:
        concepts_str: Comma-separated string of concepts (e.g., "person,car,dog")

    Returns:
        List of concept strings, stripped and lowercased
    """
    if not concepts_str:
        return []

    concepts = [c.strip().lower() for c in concepts_str.split(",")]
    return [c for c in concepts if c]


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename.

    Args:
        name: Input string

    Returns:
        Sanitized string safe for filenames
    """
    sanitized = re.sub(r"[^\w\-]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_")
    return sanitized.lower()


def get_output_filename(
    input_path: Path,
    task: str,
    suffix: str,
    output_dir: Path,
    instance_id: Optional[int] = None,
) -> Path:
    """Generate output filename for processed video.

    Args:
        input_path: Original video file path
        task: Task name (e.g., "segment", "depth", "matte")
        suffix: Additional suffix (e.g., "person", "v2")
        output_dir: Output directory
        instance_id: Optional instance number

    Returns:
        Path for the output video file
    """
    stem = input_path.stem
    safe_suffix = sanitize_filename(suffix)

    if instance_id is not None:
        filename = f"{stem}_{task}_{safe_suffix}_{instance_id:03d}.mp4"
    else:
        filename = f"{stem}_{task}_{safe_suffix}.mp4"

    return output_dir / filename


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)


def get_video_info(video_path: Path) -> dict:
    """Get basic video information using OpenCV.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video info (width, height, fps, frame_count, duration)
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoNotFoundError(f"Cannot open video: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "duration_str": format_duration(duration),
        }
    finally:
        cap.release()


# Available models registry
AVAILABLE_MODELS = {
    "segment": {
        "sam2": {"name": "SAM 2", "desc": "Meta's Segment Anything 2"},
        "grounded": {"name": "Grounded SAM 2", "desc": "Text-prompted segmentation"},
        "deva": {"name": "DEVA", "desc": "Decoupled video segmentation"},
    },
    "depth": {
        "video-depth": {"name": "Video Depth Anything", "desc": "Consistent depth for long videos"},
        "depth-v2": {"name": "Depth Anything V2", "desc": "Fast monocular depth"},
    },
    "matting": {
        "rvm": {"name": "RobustVideoMatting", "desc": "Real-time background removal"},
        "modnet": {"name": "MODNet", "desc": "Portrait matting"},
    },
    "inpaint": {
        "propainter": {"name": "ProPainter", "desc": "Object removal with flow completion"},
        "e2fgvi": {"name": "E2FGVI", "desc": "Flow-guided video inpainting"},
    },
    "upscale": {
        "realesrgan": {"name": "Real-ESRGAN", "desc": "Video/image upscaling with anime support"},
        "video2x": {"name": "Video2X", "desc": "Multi-backend upscaling framework"},
    },
    "interpolate": {
        "rife": {"name": "RIFE", "desc": "Real-time frame interpolation (4K@30fps)"},
        "film": {"name": "FILM", "desc": "Frame interpolation for large motion"},
    },
    "face": {
        "gfpgan": {"name": "GFPGAN", "desc": "GAN-based face restoration"},
        "codeformer": {"name": "CodeFormer", "desc": "Robust blind face restoration"},
    },
    "flow": {
        "raft": {"name": "RAFT", "desc": "Recurrent all-pairs field transforms"},
        "unimatch": {"name": "UniMatch", "desc": "Unified flow/depth/stereo matching"},
    },
    "stabilize": {
        "deepstab": {"name": "DeepStab", "desc": "Deep learning video stabilization"},
    },
    "denoise": {
        "fastdvdnet": {"name": "FastDVDnet", "desc": "Real-time deep video denoising"},
        "videnn": {"name": "ViDeNN", "desc": "Blind denoising with low-light support"},
    },
    "colorize": {
        "deoldify": {"name": "DeOldify", "desc": "NoGAN video colorization"},
    },
    "generate": {
        "wan": {"name": "Wan 2.1", "desc": "Alibaba text/image-to-video (MoE)"},
        "cogvideo": {"name": "CogVideoX", "desc": "Zhipu AI text/image-to-video"},
    },
    "style": {
        "adain": {"name": "AdaIN", "desc": "Real-time arbitrary style transfer"},
        "fast-artistic": {"name": "Fast Artistic", "desc": "Pre-trained styles with temporal consistency"},
    },
}


def list_models() -> str:
    """Get formatted list of available models.

    Returns:
        Formatted string with model information
    """
    lines = ["Available Models", "=" * 50, ""]

    for category, models in AVAILABLE_MODELS.items():
        lines.append(f"{category.upper()}")
        for model_id, info in models.items():
            lines.append(f"  {model_id:<15} {info['name']} - {info['desc']}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Batch Processing Utilities
# =============================================================================

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".flv"}


class BatchInput:
    """Resolve batch input specifications to video file paths.

    Supports multiple input formats:
        - Single file: "video.mp4"
        - Directory: "./videos/"
        - Glob pattern: "*.mp4" or "**/*.mp4"
        - File list: "@filelist.txt" (one path per line)
    """

    def __init__(self, input_spec: str):
        """Initialize batch input resolver.

        Args:
            input_spec: Input specification (file, directory, glob, or @filelist)
        """
        self.input_spec = input_spec
        self._paths: Optional[list[Path]] = None

    def resolve(self) -> list[Path]:
        """Resolve input specification to list of video paths.

        Returns:
            List of Path objects for video files

        Raises:
            VideoNotFoundError: If no videos found or file list doesn't exist
        """
        if self._paths is not None:
            return self._paths

        input_spec = self.input_spec

        # File list (starts with @)
        if input_spec.startswith("@"):
            file_list = Path(input_spec[1:])
            if not file_list.exists():
                raise VideoNotFoundError(f"File list not found: {file_list}")
            with open(file_list) as f:
                paths = [Path(line.strip()) for line in f if line.strip() and not line.startswith("#")]
            self._paths = [p for p in paths if p.exists() and self._is_video(p)]
            return self._paths

        path = Path(input_spec)

        # Single file
        if path.is_file():
            if not self._is_video(path):
                raise VideoNotFoundError(f"Not a video file: {path}")
            self._paths = [path]
            return self._paths

        # Directory - find all videos
        if path.is_dir():
            self._paths = sorted(self._find_videos_in_dir(path))
            return self._paths

        # Glob pattern
        if "*" in input_spec or "?" in input_spec:
            matches = glob_module.glob(input_spec, recursive=True)
            self._paths = sorted([Path(m) for m in matches if self._is_video(Path(m))])
            return self._paths

        raise VideoNotFoundError(f"Cannot resolve input: {input_spec}")

    def _is_video(self, path: Path) -> bool:
        """Check if path is a video file."""
        return path.suffix.lower() in VIDEO_EXTENSIONS

    def _find_videos_in_dir(self, directory: Path) -> Iterator[Path]:
        """Find all video files in directory (non-recursive)."""
        for ext in VIDEO_EXTENSIONS:
            yield from directory.glob(f"*{ext}")
            yield from directory.glob(f"*{ext.upper()}")

    def __len__(self) -> int:
        return len(self.resolve())

    def __iter__(self) -> Iterator[Path]:
        return iter(self.resolve())

    def __bool__(self) -> bool:
        return len(self.resolve()) > 0


class BatchResult:
    """Aggregate results from batch processing."""

    def __init__(self, task: str, input_spec: str):
        """Initialize batch result container.

        Args:
            task: Name of the processing task
            input_spec: Original input specification
        """
        self.task = task
        self.input_spec = input_spec
        self.results: list = []
        self.errors: list[tuple[Path, Exception]] = []

    def add_success(self, result) -> None:
        """Add a successful processing result."""
        self.results.append(result)

    def add_error(self, video_path: Path, error: Exception) -> None:
        """Add a failed processing record."""
        self.errors.append((video_path, error))

    @property
    def success_count(self) -> int:
        return len(self.results)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def total_count(self) -> int:
        return self.success_count + self.error_count

    def save_summary(self, output_dir: Path) -> Path:
        """Save batch processing summary to JSON.

        Args:
            output_dir: Output directory for summary file

        Returns:
            Path to saved summary file
        """
        import json
        from datetime import datetime

        summary = {
            "task": self.task,
            "input_spec": self.input_spec,
            "timestamp": datetime.now().isoformat(),
            "total": self.total_count,
            "success": self.success_count,
            "failed": self.error_count,
            "errors": [
                {"video": str(path), "error": str(err)}
                for path, err in self.errors
            ],
        }

        summary_path = output_dir / f"batch_{self.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary_path


def run_batch(
    batch_input: BatchInput,
    process_fn: Callable[[Path], "ProcessingResult"],
    task_name: str,
    parallel: int = 1,
    continue_on_error: bool = True,
) -> BatchResult:
    """Run batch processing with progress tracking.

    Args:
        batch_input: BatchInput instance with resolved paths
        process_fn: Function to process a single video (Path -> ProcessingResult)
        task_name: Name for progress display
        parallel: Number of parallel workers (default: 1 = sequential)
        continue_on_error: Continue processing if one video fails

    Returns:
        BatchResult with all results and errors
    """
    from tqdm import tqdm

    videos = batch_input.resolve()
    batch_result = BatchResult(task_name, batch_input.input_spec)

    if not videos:
        return batch_result

    if parallel <= 1:
        # Sequential processing
        with tqdm(total=len(videos), desc=f"Batch {task_name}") as pbar:
            for video_path in videos:
                try:
                    result = process_fn(video_path)
                    batch_result.add_success(result)
                except Exception as e:
                    batch_result.add_error(video_path, e)
                    if not continue_on_error:
                        raise
                finally:
                    pbar.update(1)
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(process_fn, v): v for v in videos}
            with tqdm(total=len(videos), desc=f"Batch {task_name}") as pbar:
                for future in as_completed(futures):
                    video_path = futures[future]
                    try:
                        result = future.result()
                        batch_result.add_success(result)
                    except Exception as e:
                        batch_result.add_error(video_path, e)
                        if not continue_on_error:
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise
                    finally:
                        pbar.update(1)

    return batch_result
