"""Video matting (background removal) models."""

from video_toolkit.matting.rvm import RVMatter
from video_toolkit.matting.modnet import MODNetMatter

__all__ = ["RVMatter", "MODNetMatter"]
