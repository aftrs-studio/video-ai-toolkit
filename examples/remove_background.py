#!/usr/bin/env python3
"""Example: Remove background using RobustVideoMatting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from video_toolkit import Config
from video_toolkit.matting import RVMatter


def main():
    if len(sys.argv) < 2:
        print("Usage: python remove_background.py <video>")
        print("Example: python remove_background.py video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    config = Config.from_env()
    matter = RVMatter(config)

    print(f"Removing background: {video_path}")

    # Green screen by default
    result = matter.remove_background(video_path, background_color=(0, 255, 0))

    print(f"\nCreated matted video:")
    for out in result.outputs:
        print(f"  - {out['output_path']}")


if __name__ == "__main__":
    main()
