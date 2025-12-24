"""Base processor interface for pipeline system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from video_toolkit.config import Config
    from video_toolkit.video_io import ProcessingResult


@dataclass
class ProcessorInfo:
    """Information about a processor."""

    processor_id: str
    name: str
    category: str
    description: str = ""
    default_params: dict = field(default_factory=dict)


class BaseProcessor(ABC):
    """Universal interface for all video processors.

    All processors should implement this interface to be usable in pipelines.
    """

    # Class attributes - subclasses must define these
    PROCESSOR_ID: str = ""
    PROCESSOR_NAME: str = ""
    CATEGORY: str = ""  # enhancement, analysis, creative, composition, generation
    DESCRIPTION: str = ""

    def __init__(self, config: Optional["Config"] = None):
        """Initialize processor with configuration.

        Args:
            config: Configuration object
        """
        from video_toolkit.config import Config
        self.config = config or Config.from_env()

    @abstractmethod
    def process(self, video_path: Path, **kwargs) -> "ProcessingResult":
        """Process a video file.

        Args:
            video_path: Path to input video
            **kwargs: Processor-specific parameters

        Returns:
            ProcessingResult with output paths
        """
        pass

    @classmethod
    def get_default_params(cls) -> dict[str, Any]:
        """Return default parameters for this processor.

        Returns:
            Dictionary of parameter names to default values
        """
        return {}

    @classmethod
    def get_param_schema(cls) -> dict[str, Any]:
        """Return JSON schema for parameters.

        Returns:
            JSON schema dict describing parameters
        """
        return {"type": "object", "properties": {}}

    @classmethod
    def get_info(cls) -> ProcessorInfo:
        """Get processor information.

        Returns:
            ProcessorInfo dataclass
        """
        return ProcessorInfo(
            processor_id=cls.PROCESSOR_ID,
            name=cls.PROCESSOR_NAME,
            category=cls.CATEGORY,
            description=cls.DESCRIPTION,
            default_params=cls.get_default_params(),
        )


# Processor category constants
CATEGORY_ENHANCEMENT = "enhancement"  # upscale, denoise, stabilize, face, interpolate
CATEGORY_ANALYSIS = "analysis"  # depth, flow, segment
CATEGORY_CREATIVE = "creative"  # style, colorize
CATEGORY_COMPOSITION = "composition"  # matting, inpaint
CATEGORY_GENERATION = "generation"  # generate (wan, cogvideo)

ALL_CATEGORIES = [
    CATEGORY_ENHANCEMENT,
    CATEGORY_ANALYSIS,
    CATEGORY_CREATIVE,
    CATEGORY_COMPOSITION,
    CATEGORY_GENERATION,
]
