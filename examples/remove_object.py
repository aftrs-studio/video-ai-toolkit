#!/usr/bin/env python3
"""Example: Remove objects using ProPainter inpainting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from video_toolkit import Config
from video_toolkit.inpaint import ProPainterInpainter


def main():
    if len(sys.argv) < 3:
        print("Usage: python remove_object.py <video> <mask>")
        print("Example: python remove_object.py video.mp4 mask.mp4")
        print("\nMask should be white where you want to remove objects.")
        sys.exit(1)

    video_path = sys.argv[1]
    mask_path = sys.argv[2]

    config = Config.from_env()
    inpainter = ProPainterInpainter(config)

    print(f"Inpainting video: {video_path}")
    print(f"Using mask: {mask_path}")

    result = inpainter.inpaint_video(video_path, mask_path)

    print(f"\nCreated inpainted video:")
    for out in result.outputs:
        print(f"  - {out['output_path']}")


if __name__ == "__main__":
    main()
