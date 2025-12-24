"""Processor adapters for pipeline integration.

Wraps existing video_toolkit processors to implement BaseProcessor interface.
"""

from pathlib import Path
from typing import Any, Optional

from video_toolkit.config import Config
from video_toolkit.video_io import ProcessingResult
from video_toolkit.pipeline.base import (
    BaseProcessor,
    CATEGORY_ENHANCEMENT,
    CATEGORY_ANALYSIS,
    CATEGORY_CREATIVE,
    CATEGORY_COMPOSITION,
    CATEGORY_GENERATION,
)
from video_toolkit.pipeline.registry import ProcessorRegistry


# =============================================================================
# Enhancement Processors
# =============================================================================

@ProcessorRegistry.register
class UpscaleProcessor(BaseProcessor):
    """Upscale video using Real-ESRGAN."""

    PROCESSOR_ID = "upscale"
    PROCESSOR_NAME = "Real-ESRGAN Upscaler"
    CATEGORY = CATEGORY_ENHANCEMENT
    DESCRIPTION = "AI video/image upscaling with anime support"

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from video_toolkit.upscale import RealESRGANUpscaler
            self._processor = RealESRGANUpscaler(self.config)
        return self._processor

    def process(self, video_path: Path, **kwargs) -> ProcessingResult:
        processor = self._get_processor()
        return processor.upscale_video(
            video_path,
            scale=kwargs.get("scale", 4),
            preset=kwargs.get("preset", "general"),
            enhance_face=kwargs.get("enhance_face", False),
        )

    @classmethod
    def get_default_params(cls) -> dict[str, Any]:
        return {"scale": 4, "preset": "general", "enhance_face": False}

    @classmethod
    def get_param_schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "scale": {"type": "integer", "enum": [2, 4], "default": 4},
                "preset": {"type": "string", "enum": ["general", "anime", "fast"], "default": "general"},
                "enhance_face": {"type": "boolean", "default": False},
            },
        }


@ProcessorRegistry.register
class DenoiseProcessor(BaseProcessor):
    """Denoise video using FastDVDnet."""

    PROCESSOR_ID = "denoise"
    PROCESSOR_NAME = "FastDVDnet Denoiser"
    CATEGORY = CATEGORY_ENHANCEMENT
    DESCRIPTION = "Real-time deep video denoising"

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from video_toolkit.denoise import FastDVDnetDenoiser
            self._processor = FastDVDnetDenoiser(self.config)
        return self._processor

    def process(self, video_path: Path, **kwargs) -> ProcessingResult:
        processor = self._get_processor()
        return processor.denoise_video(
            video_path,
            sigma=kwargs.get("sigma", 25),
        )

    @classmethod
    def get_default_params(cls) -> dict[str, Any]:
        return {"sigma": 25}

    @classmethod
    def get_param_schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sigma": {"type": "number", "minimum": 0, "maximum": 100, "default": 25},
            },
        }


@ProcessorRegistry.register
class StabilizeProcessor(BaseProcessor):
    """Stabilize video using DeepStab."""

    PROCESSOR_ID = "stabilize"
    PROCESSOR_NAME = "DeepStab Stabilizer"
    CATEGORY = CATEGORY_ENHANCEMENT
    DESCRIPTION = "Deep learning video stabilization"

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from video_toolkit.stabilize import DeepStabStabilizer
            self._processor = DeepStabStabilizer(self.config)
        return self._processor

    def process(self, video_path: Path, **kwargs) -> ProcessingResult:
        processor = self._get_processor()
        return processor.stabilize_video(video_path)

    @classmethod
    def get_default_params(cls) -> dict[str, Any]:
        return {}


@ProcessorRegistry.register
class FaceProcessor(BaseProcessor):
    """Restore faces using CodeFormer."""

    PROCESSOR_ID = "face"
    PROCESSOR_NAME = "CodeFormer Face Restorer"
    CATEGORY = CATEGORY_ENHANCEMENT
    DESCRIPTION = "Robust blind face restoration"

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from video_toolkit.face import CodeFormerRestorer
            self._processor = CodeFormerRestorer(self.config)
        return self._processor

    def process(self, video_path: Path, **kwargs) -> ProcessingResult:
        processor = self._get_processor()
        return processor.restore_video(
            video_path,
            fidelity=kwargs.get("fidelity", 0.5),
        )

    @classmethod
    def get_default_params(cls) -> dict[str, Any]:
        return {"fidelity": 0.5}

    @classmethod
    def get_param_schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "fidelity": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
            },
        }


@ProcessorRegistry.register
class InterpolateProcessor(BaseProcessor):
    """Interpolate frames using RIFE."""

    PROCESSOR_ID = "interpolate"
    PROCESSOR_NAME = "RIFE Interpolator"
    CATEGORY = CATEGORY_ENHANCEMENT
    DESCRIPTION = "Real-time frame interpolation (4K@30fps)"

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from video_toolkit.interpolate import RIFEInterpolator
            self._processor = RIFEInterpolator(self.config)
        return self._processor

    def process(self, video_path: Path, **kwargs) -> ProcessingResult:
        processor = self._get_processor()
        return processor.interpolate_video(
            video_path,
            multiplier=kwargs.get("multiplier", 2),
        )

    @classmethod
    def get_default_params(cls) -> dict[str, Any]:
        return {"multiplier": 2}

    @classmethod
    def get_param_schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "multiplier": {"type": "integer", "enum": [2, 4, 8], "default": 2},
            },
        }


# =============================================================================
# Analysis Processors
# =============================================================================

@ProcessorRegistry.register
class DepthProcessor(BaseProcessor):
    """Estimate depth using Video Depth Anything."""

    PROCESSOR_ID = "depth"
    PROCESSOR_NAME = "Video Depth Anything"
    CATEGORY = CATEGORY_ANALYSIS
    DESCRIPTION = "Consistent depth estimation for long videos"

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from video_toolkit.depth import VideoDepthEstimator
            self._processor = VideoDepthEstimator(self.config)
        return self._processor

    def process(self, video_path: Path, **kwargs) -> ProcessingResult:
        processor = self._get_processor()
        return processor.estimate_depth(
            video_path,
            variant=kwargs.get("variant", "large"),
        )

    @classmethod
    def get_default_params(cls) -> dict[str, Any]:
        return {"variant": "large"}

    @classmethod
    def get_param_schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "variant": {"type": "string", "enum": ["small", "base", "large"], "default": "large"},
            },
        }


@ProcessorRegistry.register
class FlowProcessor(BaseProcessor):
    """Estimate optical flow using RAFT."""

    PROCESSOR_ID = "flow"
    PROCESSOR_NAME = "RAFT Optical Flow"
    CATEGORY = CATEGORY_ANALYSIS
    DESCRIPTION = "Recurrent all-pairs field transforms"

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from video_toolkit.flow import RAFTFlowEstimator
            self._processor = RAFTFlowEstimator(self.config)
        return self._processor

    def process(self, video_path: Path, **kwargs) -> ProcessingResult:
        processor = self._get_processor()
        return processor.estimate_flow(video_path)

    @classmethod
    def get_default_params(cls) -> dict[str, Any]:
        return {}


@ProcessorRegistry.register
class SegmentProcessor(BaseProcessor):
    """Segment video using SAM2."""

    PROCESSOR_ID = "segment"
    PROCESSOR_NAME = "SAM 2 Segmenter"
    CATEGORY = CATEGORY_ANALYSIS
    DESCRIPTION = "Meta's Segment Anything 2"

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from video_toolkit.segment import SAM2Segmenter
            self._processor = SAM2Segmenter(self.config)
        return self._processor

    def process(self, video_path: Path, **kwargs) -> ProcessingResult:
        processor = self._get_processor()
        concepts = kwargs.get("concepts", [])
        if isinstance(concepts, str):
            concepts = [c.strip() for c in concepts.split(",")]
        return processor.segment_video(
            video_path,
            concepts=concepts,
        )

    @classmethod
    def get_default_params(cls) -> dict[str, Any]:
        return {"concepts": []}

    @classmethod
    def get_param_schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "concepts": {
                    "oneOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "string"},
                    ],
                    "default": [],
                },
            },
        }


# =============================================================================
# Creative Processors
# =============================================================================

@ProcessorRegistry.register
class StyleProcessor(BaseProcessor):
    """Apply style transfer using AdaIN."""

    PROCESSOR_ID = "style"
    PROCESSOR_NAME = "AdaIN Style Transfer"
    CATEGORY = CATEGORY_CREATIVE
    DESCRIPTION = "Real-time arbitrary style transfer"

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from video_toolkit.style import AdaINStyler
            self._processor = AdaINStyler(self.config)
        return self._processor

    def process(self, video_path: Path, **kwargs) -> ProcessingResult:
        processor = self._get_processor()
        style_image = kwargs.get("style", kwargs.get("style_image"))
        if not style_image:
            raise ValueError("style parameter required (path to style image)")
        return processor.stylize_video(
            video_path,
            style_image=style_image,
            alpha=kwargs.get("alpha", 1.0),
            preserve_color=kwargs.get("preserve_color", False),
        )

    @classmethod
    def get_default_params(cls) -> dict[str, Any]:
        return {"style": "", "alpha": 1.0, "preserve_color": False}

    @classmethod
    def get_param_schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "style": {"type": "string", "description": "Path to style image"},
                "alpha": {"type": "number", "minimum": 0, "maximum": 1, "default": 1.0},
                "preserve_color": {"type": "boolean", "default": False},
            },
            "required": ["style"],
        }


@ProcessorRegistry.register
class ColorizeProcessor(BaseProcessor):
    """Colorize video using DeOldify."""

    PROCESSOR_ID = "colorize"
    PROCESSOR_NAME = "DeOldify Colorizer"
    CATEGORY = CATEGORY_CREATIVE
    DESCRIPTION = "NoGAN video colorization"

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from video_toolkit.colorize import DeOldifyColorizer
            self._processor = DeOldifyColorizer(self.config)
        return self._processor

    def process(self, video_path: Path, **kwargs) -> ProcessingResult:
        processor = self._get_processor()
        return processor.colorize_video(
            video_path,
            render_factor=kwargs.get("render_factor", 35),
        )

    @classmethod
    def get_default_params(cls) -> dict[str, Any]:
        return {"render_factor": 35}

    @classmethod
    def get_param_schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "render_factor": {"type": "integer", "minimum": 7, "maximum": 45, "default": 35},
            },
        }


# =============================================================================
# Composition Processors
# =============================================================================

@ProcessorRegistry.register
class MatteProcessor(BaseProcessor):
    """Extract matte using RobustVideoMatting."""

    PROCESSOR_ID = "matte"
    PROCESSOR_NAME = "RobustVideoMatting"
    CATEGORY = CATEGORY_COMPOSITION
    DESCRIPTION = "Real-time background removal"

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from video_toolkit.matting import RVMMatter
            self._processor = RVMMatter(self.config)
        return self._processor

    def process(self, video_path: Path, **kwargs) -> ProcessingResult:
        processor = self._get_processor()
        return processor.extract_matte(
            video_path,
            output_type=kwargs.get("output_type", "alpha"),
        )

    @classmethod
    def get_default_params(cls) -> dict[str, Any]:
        return {"output_type": "alpha"}

    @classmethod
    def get_param_schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "output_type": {"type": "string", "enum": ["alpha", "composite", "foreground"], "default": "alpha"},
            },
        }


@ProcessorRegistry.register
class InpaintProcessor(BaseProcessor):
    """Inpaint video using ProPainter."""

    PROCESSOR_ID = "inpaint"
    PROCESSOR_NAME = "ProPainter Inpainter"
    CATEGORY = CATEGORY_COMPOSITION
    DESCRIPTION = "Object removal with flow completion"

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from video_toolkit.inpaint import ProPainterInpainter
            self._processor = ProPainterInpainter(self.config)
        return self._processor

    def process(self, video_path: Path, **kwargs) -> ProcessingResult:
        processor = self._get_processor()
        mask_path = kwargs.get("mask", kwargs.get("mask_path"))
        if not mask_path:
            raise ValueError("mask parameter required (path to mask video)")
        return processor.inpaint_video(
            video_path,
            mask_path=mask_path,
        )

    @classmethod
    def get_default_params(cls) -> dict[str, Any]:
        return {"mask": ""}

    @classmethod
    def get_param_schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mask": {"type": "string", "description": "Path to mask video"},
            },
            "required": ["mask"],
        }


# =============================================================================
# Generation Processors
# =============================================================================

@ProcessorRegistry.register
class GenerateProcessor(BaseProcessor):
    """Generate video using Wan 2.1."""

    PROCESSOR_ID = "generate"
    PROCESSOR_NAME = "Wan 2.1 Generator"
    CATEGORY = CATEGORY_GENERATION
    DESCRIPTION = "Alibaba text/image-to-video (MoE)"

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from video_toolkit.generate import WanGenerator
            self._processor = WanGenerator(self.config)
        return self._processor

    def process(self, video_path: Path, **kwargs) -> ProcessingResult:
        processor = self._get_processor()
        prompt = kwargs.get("prompt", "")
        if not prompt:
            # Image-to-video mode
            return processor.image_to_video(
                video_path,  # Using first frame as image
                prompt=kwargs.get("motion_prompt", "slow motion"),
            )
        else:
            # Text-to-video mode (video_path ignored)
            return processor.text_to_video(
                prompt=prompt,
                negative_prompt=kwargs.get("negative_prompt", ""),
            )

    @classmethod
    def get_default_params(cls) -> dict[str, Any]:
        return {"prompt": "", "motion_prompt": "slow motion", "negative_prompt": ""}

    @classmethod
    def get_param_schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Text prompt for video generation"},
                "motion_prompt": {"type": "string", "default": "slow motion"},
                "negative_prompt": {"type": "string", "default": ""},
            },
        }
