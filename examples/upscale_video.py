#!/usr/bin/env python3
"""Example: Upscale video using Real-ESRGAN."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from video_toolkit import Config
from video_toolkit.upscale import RealESRGANUpscaler


def main():
    if len(sys.argv) < 2:
        print("Usage: python upscale_video.py <video> [scale] [preset]")
        print("Example: python upscale_video.py video.mp4 4 general")
        print("\nPresets: general, anime, fast")
        sys.exit(1)

    video_path = sys.argv[1]
    scale = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    preset = sys.argv[3] if len(sys.argv) > 3 else "general"

    config = Config.from_env()
    upscaler = RealESRGANUpscaler(config)

    print(f"Upscaling video: {video_path}")
    print(f"Scale: {scale}x, Preset: {preset}")

    result = upscaler.upscale_video(video_path, scale, preset)

    print(f"\nCreated upscaled video:")
    for out in result.outputs:
        print(f"  - {out['output_path']}")


if __name__ == "__main__":
    main()
