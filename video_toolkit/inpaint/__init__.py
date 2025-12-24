"""Video inpainting (object removal) models."""

from video_toolkit.inpaint.propainter import ProPainterInpainter
from video_toolkit.inpaint.e2fgvi import E2FGVIInpainter

__all__ = ["ProPainterInpainter", "E2FGVIInpainter"]
