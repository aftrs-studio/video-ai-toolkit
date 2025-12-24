#!/usr/bin/env python3
"""Example: Segment objects using text prompts with Grounded SAM 2."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from video_toolkit import Config
from video_toolkit.segment import GroundedSAM2Segmenter


def main():
    if len(sys.argv) < 2:
        print("Usage: python segment_with_text.py <video> [concepts]")
        print("Example: python segment_with_text.py video.mp4 'person,car'")
        sys.exit(1)

    video_path = sys.argv[1]
    concepts = sys.argv[2] if len(sys.argv) > 2 else "person"

    config = Config.from_env()
    segmenter = GroundedSAM2Segmenter(config)

    print(f"Segmenting: {video_path}")
    print(f"Concepts: {concepts}")

    result = segmenter.segment_video(video_path, concepts)

    print(f"\nCreated {len(result.outputs)} videos:")
    for out in result.outputs:
        print(f"  - {out['output_path']}")


if __name__ == "__main__":
    main()
