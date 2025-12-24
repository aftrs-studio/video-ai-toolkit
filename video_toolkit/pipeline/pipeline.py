"""Pipeline executor for chaining video processors."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from video_toolkit.pipeline.step import PipelineStep, parse_steps_string
from video_toolkit.pipeline.registry import ProcessorRegistry

if TYPE_CHECKING:
    from video_toolkit.config import Config
    from video_toolkit.video_io import ProcessingResult


@dataclass
class StepResult:
    """Result from a single pipeline step."""

    step: PipelineStep
    output_path: Path
    success: bool
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class PipelineResult:
    """Result from running a complete pipeline."""

    pipeline_name: str
    input_video: Path
    steps: list[PipelineStep]
    step_results: list[StepResult] = field(default_factory=list)
    final_output: Optional[Path] = None
    total_duration: float = 0.0
    success: bool = True

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def completed_count(self) -> int:
        return sum(1 for r in self.step_results if r.success)

    def describe(self) -> str:
        """Get human-readable description of pipeline execution."""
        step_names = " → ".join(s.display_name for s in self.steps)
        return f"{step_names} ({self.step_count} steps)"

    def save_summary(self, output_dir: Path) -> Path:
        """Save pipeline execution summary to JSON.

        Args:
            output_dir: Output directory

        Returns:
            Path to summary file
        """
        summary = {
            "pipeline": self.pipeline_name,
            "input": str(self.input_video),
            "final_output": str(self.final_output) if self.final_output else None,
            "success": self.success,
            "total_duration": self.total_duration,
            "timestamp": datetime.now().isoformat(),
            "steps": [
                {
                    "processor": r.step.processor_id,
                    "params": r.step.params,
                    "output": str(r.output_path),
                    "success": r.success,
                    "error": r.error,
                    "duration": r.duration_seconds,
                }
                for r in self.step_results
            ],
        }

        summary_path = output_dir / f"pipeline_{self.pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary_path


class Pipeline:
    """Chainable video processing pipeline.

    Build pipelines with fluent API:
        pipeline = Pipeline("enhance").add("denoise").add("upscale", scale=4).add("style", style="art.jpg")
        result = pipeline.run("input.mp4")

    Or load from config:
        pipeline = Pipeline.load("pipeline.yaml")
        result = pipeline.run("input.mp4")
    """

    def __init__(
        self,
        name: str = "pipeline",
        description: str = "",
        config: Optional["Config"] = None,
    ):
        """Initialize pipeline.

        Args:
            name: Pipeline name for identification
            description: Human-readable description
            config: Configuration object
        """
        self.name = name
        self.description = description
        self._config = config
        self.steps: list[PipelineStep] = []
        self.settings = {
            "continue_on_error": False,
            "save_intermediates": True,
        }

    @property
    def config(self) -> "Config":
        """Get configuration, creating default if needed."""
        if self._config is None:
            from video_toolkit.config import Config
            self._config = Config.from_env()
        return self._config

    def add(self, processor_id: str, **params) -> "Pipeline":
        """Add a processing step to the pipeline.

        Args:
            processor_id: Processor identifier
            **params: Processor parameters

        Returns:
            Self for chaining
        """
        step = PipelineStep(processor_id=processor_id, params=params)
        self.steps.append(step)
        return self

    def add_step(self, step: PipelineStep) -> "Pipeline":
        """Add a pre-configured step.

        Args:
            step: PipelineStep instance

        Returns:
            Self for chaining
        """
        self.steps.append(step)
        return self

    def clear(self) -> "Pipeline":
        """Remove all steps.

        Returns:
            Self for chaining
        """
        self.steps.clear()
        return self

    def run(
        self,
        video_path: Path | str,
        output_dir: Optional[Path] = None,
    ) -> PipelineResult:
        """Execute the pipeline on a video.

        Args:
            video_path: Path to input video
            output_dir: Output directory (uses config default if None)

        Returns:
            PipelineResult with all step results
        """
        import time

        video_path = Path(video_path)
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        result = PipelineResult(
            pipeline_name=self.name,
            input_video=video_path,
            steps=self.steps.copy(),
        )

        if not self.steps:
            print("Pipeline has no steps")
            return result

        print(f"Pipeline: {self.describe()}")
        print(f"Input: {video_path}")
        print("-" * 50)

        start_time = time.time()
        current_input = video_path

        for i, step in enumerate(self.steps, 1):
            step_start = time.time()
            print(f"\n[{i}/{len(self.steps)}] {step.display_name}")

            try:
                # Get processor class and instantiate
                processor_class = ProcessorRegistry.get(step.processor_id)
                processor = processor_class(self.config)

                # Run processing
                proc_result: ProcessingResult = processor.process(current_input, **step.params)

                # Get output path from result
                if proc_result.outputs:
                    output_path = Path(proc_result.outputs[0]["output_path"])
                else:
                    raise RuntimeError(f"Processor {step.processor_id} produced no output")

                step_duration = time.time() - step_start
                result.step_results.append(StepResult(
                    step=step,
                    output_path=output_path,
                    success=True,
                    duration_seconds=step_duration,
                ))

                # Use output as next input
                current_input = output_path
                print(f"  Output: {output_path.name} ({step_duration:.1f}s)")

            except Exception as e:
                step_duration = time.time() - step_start
                result.step_results.append(StepResult(
                    step=step,
                    output_path=Path(""),
                    success=False,
                    error=str(e),
                    duration_seconds=step_duration,
                ))
                result.success = False
                print(f"  ERROR: {e}")

                if not self.settings["continue_on_error"]:
                    break

        result.total_duration = time.time() - start_time
        result.final_output = current_input if result.success else None

        print("-" * 50)
        print(f"Pipeline completed: {result.completed_count}/{result.step_count} steps")
        print(f"Total time: {result.total_duration:.1f}s")
        if result.final_output:
            print(f"Final output: {result.final_output}")

        # Save summary
        result.save_summary(output_dir)

        return result

    def describe(self) -> str:
        """Get human-readable description of pipeline."""
        step_names = " → ".join(s.display_name for s in self.steps)
        return f"{step_names} ({len(self.steps)} steps)"

    def to_dict(self) -> dict[str, Any]:
        """Serialize pipeline to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: Optional["Config"] = None) -> "Pipeline":
        """Create pipeline from dictionary.

        Args:
            data: Dictionary with name, steps, settings
            config: Configuration object

        Returns:
            Pipeline instance
        """
        pipeline = cls(
            name=data.get("name", "pipeline"),
            description=data.get("description", ""),
            config=config,
        )

        for step_data in data.get("steps", []):
            pipeline.add_step(PipelineStep.from_dict(step_data))

        if "settings" in data:
            pipeline.settings.update(data["settings"])

        return pipeline

    @classmethod
    def from_steps_string(cls, steps_str: str, name: str = "inline", config: Optional["Config"] = None) -> "Pipeline":
        """Create pipeline from inline steps string.

        Args:
            steps_str: Steps specification (e.g., "denoise,upscale:scale=4")
            name: Pipeline name
            config: Configuration object

        Returns:
            Pipeline instance
        """
        pipeline = cls(name=name, config=config)
        for step in parse_steps_string(steps_str):
            pipeline.add_step(step)
        return pipeline

    def save(self, path: Path | str) -> None:
        """Save pipeline configuration to file.

        Supports YAML and JSON formats based on extension.

        Args:
            path: Output file path (.yaml, .yml, or .json)
        """
        path = Path(path)

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                with open(path, "w") as f:
                    yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
            except ImportError:
                # Fall back to JSON
                path = path.with_suffix(".json")
                with open(path, "w") as f:
                    json.dump(self.to_dict(), f, indent=2)
        else:
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)

        print(f"Pipeline saved: {path}")

    @classmethod
    def load(cls, path: Path | str, config: Optional["Config"] = None) -> "Pipeline":
        """Load pipeline from configuration file.

        Args:
            path: Path to YAML or JSON file
            config: Configuration object

        Returns:
            Pipeline instance
        """
        path = Path(path)

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                with open(path) as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML config. Install with: pip install pyyaml")
        else:
            with open(path) as f:
                data = json.load(f)

        return cls.from_dict(data, config)
