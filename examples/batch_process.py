#!/usr/bin/env python3
"""Example: Batch process multiple videos."""

from pathlib import Path
from video_toolkit.config import Config
from video_toolkit.utils import BatchInput, run_batch
from video_toolkit.depth import VideoDepthEstimator


def main():
    config = Config.from_env()
    config.ensure_dirs()

    # Batch input options:
    # 1. Directory: "./videos/"
    # 2. Glob pattern: "*.mp4" or "**/*.mp4"
    # 3. File list: "@filelist.txt" (one path per line)

    # Example: Process all videos in a directory
    batch = BatchInput("./videos/")
    videos = batch.resolve()

    print(f"Found {len(videos)} videos")
    for v in videos:
        print(f"  - {v}")

    # Create processor
    estimator = VideoDepthEstimator(config)

    def process_one(video_path: Path):
        return estimator.estimate_depth(str(video_path), "large")

    # Run batch with parallel processing
    result = run_batch(
        batch,
        process_one,
        "depth",
        parallel=2,          # Use 2 parallel workers
        continue_on_error=True,  # Skip failures
    )

    print(f"\nCompleted: {result.success_count}/{result.total_count}")

    if result.errors:
        print(f"Failed: {result.error_count}")
        for path, err in result.errors:
            print(f"  - {path.name}: {err}")

    # Save batch summary to JSON
    summary_path = result.save_summary(config.output_dir)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
