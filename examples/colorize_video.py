#!/usr/bin/env python3
"""Example: Colorize black and white video."""

from pathlib import Path
from video_toolkit.config import Config
from video_toolkit.colorize import DeOldifyColorizer


def main():
    config = Config.from_env()
    config.ensure_dirs()

    colorizer = DeOldifyColorizer(config)

    # Colorize with standard model
    result = colorizer.colorize_video(
        "bw_footage.mp4",
        render_factor=21,  # Quality (10-40, higher = better but slower)
        artistic=False,    # Use stable colorization
    )
    print(f"Standard output: {result.outputs[0].path}")

    # Colorize with artistic model (more vibrant colors)
    result2 = colorizer.colorize_video(
        "old_film.mp4",
        render_factor=35,
        artistic=True,  # More saturated, artistic colors
    )
    print(f"Artistic output: {result2.outputs[0].path}")


if __name__ == "__main__":
    main()
