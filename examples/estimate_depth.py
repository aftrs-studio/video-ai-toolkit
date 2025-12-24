#!/usr/bin/env python3
"""Example: Estimate depth for video using Video Depth Anything."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from video_toolkit import Config
from video_toolkit.depth import VideoDepthEstimator


def main():
    if len(sys.argv) < 2:
        print("Usage: python estimate_depth.py <video>")
        print("Example: python estimate_depth.py video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    config = Config.from_env()
    estimator = VideoDepthEstimator(config)

    print(f"Estimating depth: {video_path}")

    result = estimator.estimate_depth(video_path)

    print(f"\nCreated depth video:")
    for out in result.outputs:
        print(f"  - {out['output_path']}")


if __name__ == "__main__":
    main()
