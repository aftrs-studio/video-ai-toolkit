#!/usr/bin/env python3
"""Example: Estimate optical flow using RAFT."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from video_toolkit import Config
from video_toolkit.flow import RAFTEstimator


def main():
    if len(sys.argv) < 2:
        print("Usage: python estimate_flow.py <video> [variant]")
        print("Example: python estimate_flow.py video.mp4 standard")
        print("\nVariant: small, standard (default: standard)")
        sys.exit(1)

    video_path = sys.argv[1]
    variant = sys.argv[2] if len(sys.argv) > 2 else "standard"

    config = Config.from_env()
    estimator = RAFTEstimator(config)

    print(f"Estimating optical flow: {video_path}")
    print(f"Variant: {variant}")

    result = estimator.estimate_flow(video_path, variant)

    print(f"\nCreated flow visualization:")
    for out in result.outputs:
        print(f"  - {out['output_path']}")


if __name__ == "__main__":
    main()
