#!/usr/bin/env python3
"""Example: Build and run processing pipelines."""

from pathlib import Path
from video_toolkit.config import Config
from video_toolkit.pipeline import Pipeline, ProcessorRegistry


def main():
    config = Config.from_env()
    config.ensure_dirs()

    # List available processors
    print("Available processors:")
    print(ProcessorRegistry.format_list())

    # Method 1: Build pipeline with fluent API
    pipeline = (
        Pipeline("enhance", config=config)
        .add("denoise", sigma=25)
        .add("upscale", scale=4)
    )

    print(f"\nPipeline: {pipeline.describe()}")

    # Run pipeline
    result = pipeline.run("input_video.mp4")

    print(f"\nCompleted: {result.completed_count}/{result.step_count}")
    if result.final_output:
        print(f"Final output: {result.final_output}")

    # Method 2: Build from inline string
    pipeline2 = Pipeline.from_steps_string(
        "depth,style:style=painting.jpg,upscale:scale=2",
        name="creative",
        config=config,
    )

    print(f"\nCreative pipeline: {pipeline2.describe()}")

    # Method 3: Load from config file
    # pipeline3 = Pipeline.load("pipeline.yaml", config=config)
    # result3 = pipeline3.run("input.mp4")

    # Method 4: Save pipeline for reuse
    pipeline.save("my_enhance_pipeline.yaml")
    print("\nPipeline saved to my_enhance_pipeline.yaml")


if __name__ == "__main__":
    main()
