#!/usr/bin/env python3
"""Example: Generate and run random processing pipelines."""

from video_toolkit.config import Config
from video_toolkit.pipeline import RandomPipeline, random_pipeline


def main():
    config = Config.from_env()
    config.ensure_dirs()

    # Quick one-liner random processing
    print("Running quick random pipeline...")
    result = random_pipeline(
        "input_video.mp4",
        min_steps=2,
        max_steps=4,
        seed=42,  # For reproducibility
    )
    print(f"Steps executed: {result.pipeline_name}")
    print(f"Completed: {result.completed_count}/{result.step_count}")

    # More control with RandomPipeline class
    print("\n" + "=" * 50)
    print("Creating controlled random pipeline...")

    generator = RandomPipeline(
        min_steps=2,
        max_steps=4,
        categories=["enhancement", "creative"],  # Only these categories
        exclude=["generate"],  # Skip slow generation
        config=config,
    )

    # Preview what would be generated
    print("\nPreview (seed=123):")
    print(generator.preview(seed=123))

    # List available processors for this config
    print(f"\nAvailable processors: {generator.available_processors()}")

    # Generate and run
    result = generator.run_random("input_video.mp4", seed=123)

    if result.success:
        print(f"\nSuccess! Final output: {result.final_output}")
    else:
        print(f"\nFailed at step {result.completed_count + 1}")

    # Save the summary
    summary_path = result.save_summary(config.output_dir)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
