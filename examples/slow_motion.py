#!/usr/bin/env python3
"""Example: Create slow motion video using RIFE frame interpolation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from video_toolkit import Config
from video_toolkit.interpolate import RIFEInterpolator


def main():
    if len(sys.argv) < 2:
        print("Usage: python slow_motion.py <video> [multiplier]")
        print("Example: python slow_motion.py video.mp4 4")
        print("\nMultiplier: 2, 4, or 8 (default: 2)")
        sys.exit(1)

    video_path = sys.argv[1]
    multiplier = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    config = Config.from_env()
    interpolator = RIFEInterpolator(config)

    print(f"Creating slow motion: {video_path}")
    print(f"Multiplier: {multiplier}x")

    result = interpolator.interpolate_video(video_path, multiplier)

    print(f"\nCreated interpolated video:")
    for out in result.outputs:
        print(f"  - {out['output_path']}")
        print(f"    Source FPS: {out['metadata']['source_fps']}")
        print(f"    Output FPS: {out['metadata']['output_fps']}")


if __name__ == "__main__":
    main()
