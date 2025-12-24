"""CogVideoX video generator wrapper."""

from pathlib import Path
from typing import Optional

import numpy as np

from video_toolkit.config import Config
from video_toolkit.video_io import VideoWriter, ProcessingResult
from video_toolkit.utils import ToolkitError


class CogVideoGenerator:
    """CogVideoX video generator.

    Zhipu AI's text-to-video model with 2B and 5B variants.
    Generates 6-10 second clips at 8 fps.

    Features:
        - Text-to-video generation
        - Image-to-video generation (CogVideoX-5B-I2V)
        - High fidelity output
        - Runs on 12GB+ VRAM
    """

    MODELS = {
        "2b": "THUDM/CogVideoX-2b",
        "5b": "THUDM/CogVideoX-5b",
        "5b-i2v": "THUDM/CogVideoX-5b-I2V",
    }

    def __init__(self, config: Config) -> None:
        self.config = config
        self._pipeline = None
        self._model_id = None

    def _load_model(self, variant: str = "5b") -> None:
        """Load CogVideoX model."""
        model_id = self.MODELS.get(variant, self.MODELS["5b"])

        if self._pipeline is not None and self._model_id == model_id:
            return

        try:
            import torch
            from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline
        except ImportError:
            raise ToolkitError(
                "Required packages not installed. Run: "
                "pip install torch diffusers transformers accelerate"
            )

        import torch

        device = torch.device(self.config.device)

        if "i2v" in variant:
            self._pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if self.config.device == "cuda" else torch.float32,
            )
        else:
            self._pipeline = CogVideoXPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if self.config.device == "cuda" else torch.float32,
            )

        self._pipeline = self._pipeline.to(device)
        self._model_id = model_id

        # Enable memory optimizations
        if hasattr(self._pipeline, "enable_sequential_cpu_offload"):
            self._pipeline.enable_sequential_cpu_offload()
        if hasattr(self._pipeline, "vae"):
            self._pipeline.vae.enable_slicing()
            self._pipeline.vae.enable_tiling()

    def generate_from_text(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 49,
        height: int = 480,
        width: int = 720,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        seed: Optional[int] = None,
        variant: str = "5b",
    ) -> ProcessingResult:
        """Generate video from text prompt.

        Args:
            prompt: Text description of the video to generate
            negative_prompt: What to avoid in the generation
            num_frames: Number of frames (49 = ~6 sec at 8fps)
            height: Video height
            width: Video width
            num_inference_steps: Denoising steps
            guidance_scale: How closely to follow the prompt
            seed: Random seed for reproducibility
            variant: Model variant (2b, 5b)

        Returns:
            ProcessingResult with generated video path
        """
        import torch

        self._load_model(variant)

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
        output_path = self.config.output_dir / f"generated_cogvideo_{seed or 'random'}.mp4"

        writer = VideoWriter(
            output_path,
            fps=8,  # CogVideoX generates at 8 fps
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
                    "model": f"cogvideox-{variant}",
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
        num_frames: int = 49,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        seed: Optional[int] = None,
    ) -> ProcessingResult:
        """Generate video from an input image.

        Args:
            image_path: Path to the source image
            prompt: Text prompt to guide generation
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
        self._load_model("5b-i2v")

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")

        # CogVideoX expects specific dimensions
        width, height = 720, 480
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
        output_path = self.config.output_dir / f"generated_cogvideo_i2v_{seed or 'random'}.mp4"

        writer = VideoWriter(
            output_path,
            fps=8,
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
                    "model": "cogvideox-5b-i2v",
                    "source_image": str(image_path),
                    "prompt": prompt,
                    "num_frames": num_frames,
                    "seed": seed,
                },
            )

        finally:
            writer.close()

        return result
