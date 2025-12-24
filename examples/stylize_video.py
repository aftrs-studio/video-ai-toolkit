#!/usr/bin/env python3
"""Example: Apply artistic style transfer to video."""

from pathlib import Path
from video_toolkit.config import Config
from video_toolkit.style import AdaINStyler, FastArtisticStyler


def main():
    config = Config.from_env()
    config.ensure_dirs()

    # Method 1: AdaIN - arbitrary style from any image
    adain = AdaINStyler(config)

    result = adain.stylize_video(
        "input_video.mp4",
        "starry_night.jpg",  # Any style image
        alpha=1.0,           # Style strength (0.0-1.0)
        preserve_color=False,
    )
    print(f"AdaIN output: {result.outputs[0].path}")

    # With color preservation
    result2 = adain.stylize_video(
        "input_video.mp4",
        "painting.jpg",
        alpha=0.7,           # Partial style
        preserve_color=True, # Keep original colors
    )
    print(f"AdaIN (color preserved): {result2.outputs[0].path}")

    # Method 2: Fast Artistic - pre-trained styles with temporal consistency
    fast = FastArtisticStyler(config)

    # List available styles
    print(f"Available styles: {fast.list_styles()}")

    result3 = fast.stylize_video(
        "input_video.mp4",
        "la_muse",           # Pre-trained style name
        temporal_weight=0.5, # Temporal consistency (0.0-1.0)
    )
    print(f"Fast Artistic output: {result3.outputs[0].path}")


if __name__ == "__main__":
    main()
