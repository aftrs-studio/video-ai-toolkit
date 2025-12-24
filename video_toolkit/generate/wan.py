"""Wan 2.1 video generator wrapper."""

from pathlib import Path
from typing import Optional

import numpy as np

from video_toolkit.config import Config
from video_toolkit.video_io import VideoWriter, ProcessingResult
from video_toolkit.utils import ToolkitError


class WanGenerator:
    """Wan 2.1/2.2 video generator.

    Alibaba's state-of-the-art text-to-video and image-to-video model.
    Uses Mixture-of-Experts (MoE) architecture for high quality output.

    Features:
        - Text-to-video generation
        - Image-to-video generation
        - Chinese and English text support
        - Runs on consumer GPUs (8GB+ VRAM)
    """

    MODELS = {
        "t2v-1.3b": "Wan-AI/Wan2.1-T2V-1.3B",
        "t2v-14b": "Wan-AI/Wan2.1-T2V-14B",
        "i2v-14b": "Wan-AI/Wan2.1-I2V-14B-720P",
        "i2v-14b-turbo": "Wan-AI/Wan2.1-I2V-14B-720P-Turbo",
    }

    def __init__(self, config: Config) -> None:
        self.config = config
        self._pipeline = None
        self._model_id = None

    def _load_model(self, variant: str = "t2v-1.3b") -> None:
        """Load Wan model."""
        model_id = self.MODELS.get(variant, self.MODELS["t2v-1.3b"])

        if self._pipeline is not None and self._model_id == model_id:
            return

        try:
            import torch
            from diffusers import DiffusionPipeline
        except ImportError:
            raise ToolkitError(
                "Required packages not installed. Run: "
                "pip install torch diffusers transformers accelerate"
            )

        import torch

        device = torch.device(self.config.device)

        self._pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
        )
        self._pipeline = self._pipeline.to(device)
        self._model_id = model_id

        # Enable memory optimizations
        if hasattr(self._pipeline, "enable_model_cpu_offload"):
            self._pipeline.enable_model_cpu_offload()

    def generate_from_text(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
    ) -> ProcessingResult:
        """Generate video from text prompt.

        Args:
            prompt: Text description of the video to generate
            negative_prompt: What to avoid in the generation
            num_frames: Number of frames to generate (default 81 = ~3 sec)
            height: Video height (must be divisible by 16)
            width: Video width (must be divisible by 16)
            num_inference_steps: Denoising steps (more = better quality)
            guidance_scale: How closely to follow the prompt
            seed: Random seed for reproducibility

        Returns:
            ProcessingResult with generated video path
        """
        import torch

        self._load_model("t2v-1.3b")

        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.config.device)
            generator.manual_seed(seed)

        # Generate
        output = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        # Get frames
        frames = output.frames[0]  # List of PIL Images

        # Save video
        output_path = self.config.output_dir / f"generated_wan_{seed or 'random'}.mp4"

        writer = VideoWriter(
            output_path,
            fps=24,
            width=width,
            height=height,
        )

        result = ProcessingResult(source_video=None)

        try:
            for frame in frames:
                frame_np = np.array(frame)[:, :, ::-1]  # RGB to BGR
                writer.write(frame_np)

            result.add_output(
                output_path,
                "generated",
                {
                    "model": "wan2.1",
                    "prompt": prompt,
                    "num_frames": num_frames,
                    "seed": seed,
                },
            )

        finally:
            writer.close()

        return result

    def generate_from_image(
        self,
        image_path: str | Path,
        prompt: str = "",
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
    ) -> ProcessingResult:
        """Generate video from an input image.

        Args:
            image_path: Path to the source image
            prompt: Optional text prompt to guide generation
            num_frames: Number of frames to generate
            num_inference_steps: Denoising steps
            guidance_scale: How closely to follow the prompt
            seed: Random seed for reproducibility

        Returns:
            ProcessingResult with generated video path
        """
        import torch
        from PIL import Image

        image_path = Path(image_path)
        self._load_model("i2v-14b-turbo")

        # Load image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Ensure dimensions are divisible by 16
        width = (width // 16) * 16
        height = (height // 16) * 16
        image = image.resize((width, height))

        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.config.device)
            generator.manual_seed(seed)

        # Generate
        output = self._pipeline(
            image=image,
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        # Get frames
        frames = output.frames[0]

        # Save video
        output_path = self.config.output_dir / f"generated_wan_i2v_{seed or 'random'}.mp4"

        writer = VideoWriter(
            output_path,
            fps=24,
            width=width,
            height=height,
        )

        result = ProcessingResult(source_video=image_path)

        try:
            for frame in frames:
                frame_np = np.array(frame)[:, :, ::-1]
                writer.write(frame_np)

            result.add_output(
                output_path,
                "generated",
                {
                    "model": "wan2.1-i2v",
                    "source_image": str(image_path),
                    "prompt": prompt,
                    "num_frames": num_frames,
                    "seed": seed,
                },
            )

        finally:
            writer.close()

        return result
