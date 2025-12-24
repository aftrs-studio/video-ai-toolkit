#!/usr/bin/env python3
"""Example: Restore faces in video using GFPGAN."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from video_toolkit import Config
from video_toolkit.face import GFPGANRestorer


def main():
    if len(sys.argv) < 2:
        print("Usage: python restore_faces.py <video> [upscale]")
        print("Example: python restore_faces.py video.mp4 2")
        print("\nUpscale: 1, 2, or 4 (default: 2)")
        sys.exit(1)

    video_path = sys.argv[1]
    upscale = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    config = Config.from_env()
    restorer = GFPGANRestorer(config)

    print(f"Restoring faces: {video_path}")
    print(f"Upscale: {upscale}x")

    result = restorer.restore_video(video_path, upscale)

    print(f"\nCreated restored video:")
    for out in result.outputs:
        print(f"  - {out['output_path']}")
        print(f"    Frames with faces: {out['metadata']['frames_with_faces']}")


if __name__ == "__main__":
    main()
