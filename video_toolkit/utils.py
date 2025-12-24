"""Utility functions for Video AI Toolkit."""

import re
from pathlib import Path
from typing import Optional


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
