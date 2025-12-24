"""Video processing pipeline system.

Build chainable processing pipelines with fluent API:

    from video_toolkit.pipeline import Pipeline

    # Build and run pipeline
    result = (
        Pipeline("enhance")
        .add("denoise", sigma=25)
        .add("upscale", scale=4)
        .add("style", style="painting.jpg")
        .run("input.mp4")
    )

    # Load from config file
    pipeline = Pipeline.load("pipeline.yaml")
    result = pipeline.run("input.mp4")

Random pipeline generation:

    from video_toolkit.pipeline import RandomPipeline, random_pipeline

    # Quick random processing
    result = random_pipeline("input.mp4", min_steps=2, max_steps=4, seed=42)

    # Or with more control
    generator = RandomPipeline(
        categories=["enhancement", "creative"],
        exclude=["generate"],
    )
    result = generator.run_random("input.mp4")

List available processors:

    from video_toolkit.pipeline import ProcessorRegistry

    # All processors
    print(ProcessorRegistry.format_list())

    # By category
    print(ProcessorRegistry.format_list("enhancement"))
"""

from video_toolkit.pipeline.base import (
    BaseProcessor,
    ProcessorInfo,
    CATEGORY_ENHANCEMENT,
    CATEGORY_ANALYSIS,
    CATEGORY_CREATIVE,
    CATEGORY_COMPOSITION,
    CATEGORY_GENERATION,
    ALL_CATEGORIES,
)
from video_toolkit.pipeline.step import PipelineStep, parse_steps_string
from video_toolkit.pipeline.registry import ProcessorRegistry
from video_toolkit.pipeline.pipeline import Pipeline, PipelineResult, StepResult
from video_toolkit.pipeline.random import RandomPipeline, random_pipeline

__all__ = [
    # Base
    "BaseProcessor",
    "ProcessorInfo",
    # Categories
    "CATEGORY_ENHANCEMENT",
    "CATEGORY_ANALYSIS",
    "CATEGORY_CREATIVE",
    "CATEGORY_COMPOSITION",
    "CATEGORY_GENERATION",
    "ALL_CATEGORIES",
    # Step
    "PipelineStep",
    "parse_steps_string",
    # Registry
    "ProcessorRegistry",
    # Pipeline
    "Pipeline",
    "PipelineResult",
    "StepResult",
    # Random
    "RandomPipeline",
    "random_pipeline",
]
