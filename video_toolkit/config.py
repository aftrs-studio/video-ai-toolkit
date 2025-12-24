"""Configuration management for Video AI Toolkit."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    """Configuration for Video AI Toolkit."""

    # HuggingFace token for model download
    hf_token: Optional[str] = None

    # Model cache path
    model_path: Path = field(default_factory=lambda: Path.home() / ".cache" / "vidtool")

    # Device for inference (cuda or cpu)
    device: str = "cuda"

    # Default output directory
    output_dir: Path = field(default_factory=lambda: Path("./output"))

    # Video processing settings
    batch_size: int = 8
    max_frames: Optional[int] = None

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        load_dotenv()

        model_path = os.getenv("VIDTOOL_MODEL_PATH")
        output_dir = os.getenv("VIDTOOL_OUTPUT_DIR")

        return cls(
            hf_token=os.getenv("HF_TOKEN"),
            model_path=Path(model_path) if model_path else Path.home() / ".cache" / "vidtool",
            device=os.getenv("VIDTOOL_DEVICE", "cuda"),
            output_dir=Path(output_dir) if output_dir else Path("./output"),
        )

    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings = []

        if not self.hf_token:
            warnings.append("HF_TOKEN not set - model download may fail")

        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    warnings.append("CUDA requested but not available - falling back to CPU")
                    self.device = "cpu"
            except ImportError:
                warnings.append("PyTorch not installed")

        return warnings

    def ensure_dirs(self) -> None:
        """Create required directories."""
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_model_dir(self, model_name: str) -> Path:
        """Get directory for a specific model's checkpoints.

        Args:
            model_name: Name of the model (e.g., "sam2", "propainter")

        Returns:
            Path to model's checkpoint directory
        """
        model_dir = self.model_path / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
