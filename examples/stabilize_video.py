#!/usr/bin/env python3
"""Example: Stabilize shaky video."""

from pathlib import Path
from video_toolkit.config import Config
from video_toolkit.stabilize import DeepStabilizer


def main():
    config = Config.from_env()
    config.ensure_dirs()

    stabilizer = DeepStabilizer(config)

    # Stabilize video with default settings
    result = stabilizer.stabilize_video(
        "input_shaky.mp4",
        smoothing=0.8,  # Higher = smoother but may lose some motion
        crop_ratio=0.9,  # Crop to hide border artifacts
    )

    print(f"Stabilized: {result.outputs[0].path}")


if __name__ == "__main__":
    main()
