"""Random pipeline generation."""

import random
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from video_toolkit.pipeline.base import BaseProcessor, ALL_CATEGORIES
from video_toolkit.pipeline.step import PipelineStep
from video_toolkit.pipeline.pipeline import Pipeline, PipelineResult
from video_toolkit.pipeline.registry import ProcessorRegistry

if TYPE_CHECKING:
    from video_toolkit.config import Config


class RandomPipeline:
    """Generate and run random processing pipelines.

    Create varied processing chains for experimentation:
        random = RandomPipeline(min_steps=2, max_steps=4)
        result = random.run_random("input.mp4", seed=42)

    Filter by category or exclude specific processors:
        random = RandomPipeline(
            categories=["enhancement", "creative"],
            exclude=["generate"],  # Skip slow generation
        )
    """

    def __init__(
        self,
        min_steps: int = 2,
        max_steps: int = 5,
        categories: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
        config: Optional["Config"] = None,
    ):
        """Initialize random pipeline generator.

        Args:
            min_steps: Minimum number of steps
            max_steps: Maximum number of steps
            categories: Only include these categories (None = all)
            exclude: Exclude these processor IDs
            config: Configuration object
        """
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.categories = categories
        self.exclude = exclude or []
        self._config = config

    @property
    def config(self) -> "Config":
        """Get configuration, creating default if needed."""
        if self._config is None:
            from video_toolkit.config import Config
            self._config = Config.from_env()
        return self._config

    def generate(self, seed: Optional[int] = None) -> Pipeline:
        """Generate a random pipeline.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Pipeline with random steps
        """
        if seed is not None:
            random.seed(seed)

        # Get candidate processors
        candidates = ProcessorRegistry.get_random_candidates(
            categories=self.categories,
            exclude=self.exclude,
        )

        if not candidates:
            raise ValueError("No processors available for random selection")

        # Determine number of steps
        num_steps = random.randint(self.min_steps, self.max_steps)
        num_steps = min(num_steps, len(candidates))

        # Select random processors (no duplicates)
        selected = random.sample(candidates, num_steps)

        # Build pipeline
        pipeline = Pipeline(name="random", config=self.config)

        for processor_class in selected:
            # Get default params with possible randomization
            params = self._randomize_params(processor_class)
            pipeline.add(processor_class.PROCESSOR_ID, **params)

        return pipeline

    def _randomize_params(self, processor_class: type[BaseProcessor]) -> dict:
        """Get randomized parameters for a processor.

        Uses defaults with possible random variations.

        Args:
            processor_class: Processor class

        Returns:
            Parameter dictionary
        """
        defaults = processor_class.get_default_params()
        params = defaults.copy()

        # Add some random variation based on processor type
        schema = processor_class.get_param_schema()
        properties = schema.get("properties", {})

        for key, prop in properties.items():
            if key not in params:
                continue

            prop_type = prop.get("type")
            if prop_type == "number" and "minimum" in prop and "maximum" in prop:
                # Random float within range
                params[key] = random.uniform(prop["minimum"], prop["maximum"])
            elif prop_type == "integer" and "minimum" in prop and "maximum" in prop:
                # Random int within range
                params[key] = random.randint(prop["minimum"], prop["maximum"])

        return params

    def run_random(
        self,
        video_path: Path | str,
        seed: Optional[int] = None,
        output_dir: Optional[Path] = None,
    ) -> PipelineResult:
        """Generate and run a random pipeline.

        Args:
            video_path: Path to input video
            seed: Random seed for reproducibility
            output_dir: Output directory

        Returns:
            PipelineResult from execution
        """
        pipeline = self.generate(seed=seed)
        print(f"Random pipeline: {pipeline.describe()}")
        return pipeline.run(video_path, output_dir=output_dir)

    def preview(self, seed: Optional[int] = None) -> str:
        """Preview what a random pipeline would look like without running.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Description of the pipeline that would be generated
        """
        pipeline = self.generate(seed=seed)
        lines = [
            f"Random Pipeline Preview (seed={seed})",
            "=" * 50,
            "",
        ]

        for i, step in enumerate(pipeline.steps, 1):
            lines.append(f"{i}. {step.processor_id}")
            if step.params:
                for k, v in step.params.items():
                    lines.append(f"     {k}: {v}")

        return "\n".join(lines)

    def available_processors(self) -> list[str]:
        """List processors available for random selection.

        Returns:
            List of processor IDs
        """
        candidates = ProcessorRegistry.get_random_candidates(
            categories=self.categories,
            exclude=self.exclude,
        )
        return sorted(c.PROCESSOR_ID for c in candidates)


# Convenience functions

def random_pipeline(
    video_path: Path | str,
    min_steps: int = 2,
    max_steps: int = 4,
    categories: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
    seed: Optional[int] = None,
) -> PipelineResult:
    """Run a random pipeline on a video.

    Convenience function for quick random processing.

    Args:
        video_path: Path to input video
        min_steps: Minimum number of steps
        max_steps: Maximum number of steps
        categories: Only include these categories
        exclude: Exclude these processor IDs
        seed: Random seed for reproducibility

    Returns:
        PipelineResult from execution
    """
    generator = RandomPipeline(
        min_steps=min_steps,
        max_steps=max_steps,
        categories=categories,
        exclude=exclude,
    )
    return generator.run_random(video_path, seed=seed)
