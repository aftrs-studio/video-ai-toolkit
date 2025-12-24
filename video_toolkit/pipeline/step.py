"""Pipeline step definition."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PipelineStep:
    """A single step in a processing pipeline.

    Represents a processor with its configuration to be executed.
    """

    processor_id: str
    params: dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None  # Custom step name for display

    def __post_init__(self):
        """Validate step after initialization."""
        if not self.processor_id:
            raise ValueError("processor_id is required")

    @property
    def display_name(self) -> str:
        """Get display name for this step."""
        return self.name or self.processor_id

    def to_dict(self) -> dict[str, Any]:
        """Serialize step to dictionary.

        Returns:
            Dictionary representation
        """
        data = {"processor": self.processor_id}
        if self.params:
            data["params"] = self.params
        if self.name:
            data["name"] = self.name
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineStep":
        """Create step from dictionary.

        Args:
            data: Dictionary with processor, params, name

        Returns:
            PipelineStep instance
        """
        return cls(
            processor_id=data["processor"],
            params=data.get("params", {}),
            name=data.get("name"),
        )

    @classmethod
    def from_string(cls, spec: str) -> "PipelineStep":
        """Parse step from string specification.

        Format: processor_id:key=value,key=value

        Examples:
            "denoise" -> PipelineStep("denoise", {})
            "upscale:scale=4" -> PipelineStep("upscale", {"scale": 4})
            "style:style=art.jpg,alpha=0.8" -> PipelineStep("style", {"style": "art.jpg", "alpha": 0.8})

        Args:
            spec: Step specification string

        Returns:
            PipelineStep instance
        """
        parts = spec.split(":", 1)
        processor_id = parts[0].strip()

        params = {}
        if len(parts) > 1 and parts[1].strip():
            for pair in parts[1].split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Try to parse as number
                    params[key] = _parse_value(value)

        return cls(processor_id=processor_id, params=params)

    def __str__(self) -> str:
        """String representation."""
        if self.params:
            param_str = ",".join(f"{k}={v}" for k, v in self.params.items())
            return f"{self.processor_id}:{param_str}"
        return self.processor_id


def _parse_value(value: str) -> Any:
    """Parse a string value to appropriate type.

    Args:
        value: String value

    Returns:
        Parsed value (int, float, bool, or string)
    """
    # Boolean
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # String
    return value


def parse_steps_string(steps_str: str) -> list[PipelineStep]:
    """Parse multiple steps from comma-separated string.

    Format: step1,step2:param=value,step3

    Args:
        steps_str: Steps specification string

    Returns:
        List of PipelineStep instances
    """
    steps = []
    # Split on comma, but not within param values
    current = ""
    depth = 0

    for char in steps_str + ",":
        if char == ":" and depth == 0:
            depth = 1
        elif char == "," and depth == 0:
            if current.strip():
                steps.append(PipelineStep.from_string(current.strip()))
            current = ""
            continue
        elif char == "," and depth == 1:
            depth = 0
            if current.strip():
                steps.append(PipelineStep.from_string(current.strip()))
            current = ""
            continue
        current += char

    return steps
