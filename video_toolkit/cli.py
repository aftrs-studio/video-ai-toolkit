"""Unified CLI for Video AI Toolkit."""

from pathlib import Path
from typing import Optional

import click

from video_toolkit import __version__
from video_toolkit.config import Config
from video_toolkit.utils import ToolkitError, list_models


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """Video AI Toolkit - Unified video processing with multiple AI models.

    Process videos with state-of-the-art AI models for segmentation,
    depth estimation, background removal, and object inpainting.

    Examples:

        vidtool segment video.mp4 -c "person,car"

        vidtool depth video.mp4

        vidtool matte video.mp4

        vidtool inpaint video.mp4 --mask mask.mp4
    """
    pass


# ============================================================================
# SEGMENT Command
# ============================================================================
@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--concepts", "-c", help="Comma-separated concepts to segment (e.g., 'person,car')")
@click.option("--model", "-m", type=click.Choice(["sam2", "grounded", "deva"]), default="grounded", help="Segmentation model")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--device", "-d", type=click.Choice(["cuda", "cpu"]), help="Device for inference")
def segment(
    video_path: str,
    concepts: Optional[str],
    model: str,
    output_dir: Optional[str],
    device: Optional[str],
) -> None:
    """Segment objects from a video.

    Uses text prompts to detect and track objects through video frames,
    outputting separate videos for each detected instance.

    Examples:

        vidtool segment video.mp4 -c "person"

        vidtool segment video.mp4 -c "person,car,dog" -m grounded

        vidtool segment video.mp4 -c "person wearing red" -o ./output
    """
    try:
        config = Config.from_env()
        if device:
            config.device = device
        if output_dir:
            config.output_dir = Path(output_dir)

        config.ensure_dirs()
        _show_warnings(config)

        click.echo(f"\nVideo AI Toolkit - Segment")
        click.echo("=" * 40)
        click.echo(f"Model: {model}")
        click.echo(f"Input: {video_path}")
        if concepts:
            click.echo(f"Concepts: {concepts}")
        click.echo("")

        if model == "grounded":
            from video_toolkit.segment import GroundedSAM2Segmenter
            segmenter = GroundedSAM2Segmenter(config)
            if not concepts:
                raise ToolkitError("Grounded SAM 2 requires concepts. Use -c 'person,car'")
            result = segmenter.segment_video(video_path, concepts)
        elif model == "sam2":
            from video_toolkit.segment import SAM2Segmenter
            segmenter = SAM2Segmenter(config)
            result = segmenter.segment_video(video_path)
        else:  # deva
            from video_toolkit.segment import DEVASegmenter
            segmenter = DEVASegmenter(config)
            result = segmenter.segment_video(video_path, concepts)

        click.echo("")
        click.secho("Segmentation complete!", fg="green")
        click.echo(f"Created {len(result.outputs)} videos in {config.output_dir}")

    except ToolkitError as e:
        click.secho(f"Error: {e}", fg="red")
        raise SystemExit(1)


# ============================================================================
# DEPTH Command
# ============================================================================
@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--model", "-m", type=click.Choice(["video-depth", "depth-v2"]), default="video-depth", help="Depth model")
@click.option("--variant", "-v", type=click.Choice(["small", "base", "large"]), default="large", help="Model variant")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--device", "-d", type=click.Choice(["cuda", "cpu"]), help="Device for inference")
def depth(
    video_path: str,
    model: str,
    variant: str,
    output_dir: Optional[str],
    device: Optional[str],
) -> None:
    """Estimate depth for video frames.

    Creates a colorized depth visualization video showing relative
    distances of objects from the camera.

    Examples:

        vidtool depth video.mp4

        vidtool depth video.mp4 -m depth-v2 -v small

        vidtool depth video.mp4 -o ./depth_output
    """
    try:
        config = Config.from_env()
        if device:
            config.device = device
        if output_dir:
            config.output_dir = Path(output_dir)

        config.ensure_dirs()
        _show_warnings(config)

        click.echo(f"\nVideo AI Toolkit - Depth")
        click.echo("=" * 40)
        click.echo(f"Model: {model} ({variant})")
        click.echo(f"Input: {video_path}")
        click.echo("")

        if model == "video-depth":
            from video_toolkit.depth import VideoDepthEstimator
            estimator = VideoDepthEstimator(config)
            result = estimator.estimate_depth(video_path, variant)
        else:  # depth-v2
            from video_toolkit.depth import DepthV2Estimator
            estimator = DepthV2Estimator(config)
            result = estimator.estimate_depth(video_path, variant)

        click.echo("")
        click.secho("Depth estimation complete!", fg="green")
        click.echo(f"Output in {config.output_dir}")

    except ToolkitError as e:
        click.secho(f"Error: {e}", fg="red")
        raise SystemExit(1)


# ============================================================================
# MATTE Command
# ============================================================================
@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--model", "-m", type=click.Choice(["rvm", "modnet"]), default="rvm", help="Matting model")
@click.option("--background", "-b", default="0,255,0", help="Background color RGB (default: green)")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--device", "-d", type=click.Choice(["cuda", "cpu"]), help="Device for inference")
def matte(
    video_path: str,
    model: str,
    background: str,
    output_dir: Optional[str],
    device: Optional[str],
) -> None:
    """Remove background from video (matting).

    Extracts foreground subjects (typically people) and replaces
    the background with a solid color.

    Examples:

        vidtool matte video.mp4

        vidtool matte video.mp4 -m modnet

        vidtool matte video.mp4 -b "0,0,255"  # Blue background
    """
    try:
        config = Config.from_env()
        if device:
            config.device = device
        if output_dir:
            config.output_dir = Path(output_dir)

        config.ensure_dirs()
        _show_warnings(config)

        # Parse background color
        bg_color = tuple(int(x) for x in background.split(","))
        if len(bg_color) != 3:
            raise ToolkitError("Background must be R,G,B format (e.g., '0,255,0')")

        click.echo(f"\nVideo AI Toolkit - Matte")
        click.echo("=" * 40)
        click.echo(f"Model: {model}")
        click.echo(f"Input: {video_path}")
        click.echo(f"Background: RGB{bg_color}")
        click.echo("")

        if model == "rvm":
            from video_toolkit.matting import RVMatter
            matter = RVMatter(config)
            result = matter.remove_background(video_path, background_color=bg_color)
        else:  # modnet
            from video_toolkit.matting import MODNetMatter
            matter = MODNetMatter(config)
            result = matter.remove_background(video_path, background_color=bg_color)

        click.echo("")
        click.secho("Background removal complete!", fg="green")
        click.echo(f"Output in {config.output_dir}")

    except ToolkitError as e:
        click.secho(f"Error: {e}", fg="red")
        raise SystemExit(1)


# ============================================================================
# INPAINT Command
# ============================================================================
@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--mask", required=True, type=click.Path(exists=True), help="Mask video (white = area to remove)")
@click.option("--model", "-m", type=click.Choice(["propainter", "e2fgvi"]), default="propainter", help="Inpainting model")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--device", "-d", type=click.Choice(["cuda", "cpu"]), help="Device for inference")
def inpaint(
    video_path: str,
    mask: str,
    model: str,
    output_dir: Optional[str],
    device: Optional[str],
) -> None:
    """Remove objects from video (inpainting).

    Uses a mask video to identify regions to remove and fills them
    with plausible content based on surrounding frames.

    Examples:

        vidtool inpaint video.mp4 --mask mask.mp4

        vidtool inpaint video.mp4 --mask mask.mp4 -m e2fgvi

        vidtool inpaint video.mp4 --mask mask.mp4 -o ./cleaned
    """
    try:
        config = Config.from_env()
        if device:
            config.device = device
        if output_dir:
            config.output_dir = Path(output_dir)

        config.ensure_dirs()
        _show_warnings(config)

        click.echo(f"\nVideo AI Toolkit - Inpaint")
        click.echo("=" * 40)
        click.echo(f"Model: {model}")
        click.echo(f"Input: {video_path}")
        click.echo(f"Mask: {mask}")
        click.echo("")

        if model == "propainter":
            from video_toolkit.inpaint import ProPainterInpainter
            inpainter = ProPainterInpainter(config)
            result = inpainter.inpaint_video(video_path, mask)
        else:  # e2fgvi
            from video_toolkit.inpaint import E2FGVIInpainter
            inpainter = E2FGVIInpainter(config)
            result = inpainter.inpaint_video(video_path, mask)

        click.echo("")
        click.secho("Inpainting complete!", fg="green")
        click.echo(f"Output in {config.output_dir}")

    except ToolkitError as e:
        click.secho(f"Error: {e}", fg="red")
        raise SystemExit(1)


# ============================================================================
# INFO Command
# ============================================================================
@cli.command()
def info() -> None:
    """Show system and GPU information."""
    try:
        import torch

        click.echo("\nVideo AI Toolkit - System Info")
        click.echo("=" * 40)
        click.echo(f"Version: {__version__}")
        click.echo(f"PyTorch: {torch.__version__}")
        click.echo(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            click.echo(f"CUDA version: {torch.version.cuda}")
            click.echo(f"GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            click.echo(f"GPU memory: {props.total_memory / 1e9:.1f} GB")

        config = Config.from_env()
        click.echo(f"\nModel cache: {config.model_path}")
        click.echo(f"Output dir: {config.output_dir}")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        raise SystemExit(1)


# ============================================================================
# MODELS Command
# ============================================================================
@cli.command()
def models() -> None:
    """List available AI models."""
    click.echo("\n" + list_models())


# ============================================================================
# Helper Functions
# ============================================================================
def _show_warnings(config: Config) -> None:
    """Show configuration warnings."""
    warnings = config.validate()
    for warning in warnings:
        click.secho(f"Warning: {warning}", fg="yellow")


if __name__ == "__main__":
    cli()
