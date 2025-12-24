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
@click.argument("video_input")
@click.option("--model", "-m", type=click.Choice(["video-depth", "depth-v2"]), default="video-depth", help="Depth model")
@click.option("--variant", "-v", type=click.Choice(["small", "base", "large"]), default="large", help="Model variant")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--device", "-d", type=click.Choice(["cuda", "cpu"]), help="Device for inference")
@click.option("--continue-on-error/--abort-on-error", default=True, help="Continue batch if one fails")
@click.option("--parallel", "-j", type=int, default=1, help="Parallel workers for batch")
def depth(
    video_input: str,
    model: str,
    variant: str,
    output_dir: Optional[str],
    device: Optional[str],
    continue_on_error: bool,
    parallel: int,
) -> None:
    """Estimate depth for video frames.

    Supports batch input: single file, directory, glob pattern, or @filelist.txt

    Examples:

        vidtool depth video.mp4

        vidtool depth ./videos/ --parallel 2

        vidtool depth "*.mp4" -m depth-v2 -v small
    """
    from video_toolkit.utils import BatchInput, run_batch

    try:
        config = Config.from_env()
        if device:
            config.device = device
        if output_dir:
            config.output_dir = Path(output_dir)

        config.ensure_dirs()
        _show_warnings(config)

        batch = BatchInput(video_input)
        videos = batch.resolve()

        if not videos:
            click.secho("No videos found matching input", fg="yellow")
            return

        click.echo(f"\nVideo AI Toolkit - Depth")
        click.echo("=" * 40)
        click.echo(f"Model: {model} ({variant})")
        click.echo(f"Videos: {len(videos)}")
        click.echo("")

        if model == "video-depth":
            from video_toolkit.depth import VideoDepthEstimator
            estimator = VideoDepthEstimator(config)
            process_fn = lambda vp: estimator.estimate_depth(str(vp), variant)
        else:  # depth-v2
            from video_toolkit.depth import DepthV2Estimator
            estimator = DepthV2Estimator(config)
            process_fn = lambda vp: estimator.estimate_depth(str(vp), variant)

        if len(videos) == 1:
            result = process_fn(videos[0])
            click.echo("")
            click.secho("Depth estimation complete!", fg="green")
        else:
            batch_result = run_batch(batch, process_fn, "depth", parallel, continue_on_error)
            click.echo("")
            click.secho(f"Batch complete: {batch_result.success_count}/{batch_result.total_count}", fg="green")
            if batch_result.errors:
                click.secho(f"Failed: {batch_result.error_count}", fg="red")
            batch_result.save_summary(config.output_dir)

        click.echo(f"Output in {config.output_dir}")

    except ToolkitError as e:
        click.secho(f"Error: {e}", fg="red")
        raise SystemExit(1)


# ============================================================================
# MATTE Command
# ============================================================================
@cli.command()
@click.argument("video_input")
@click.option("--model", "-m", type=click.Choice(["rvm", "modnet"]), default="rvm", help="Matting model")
@click.option("--background", "-b", default="0,255,0", help="Background color RGB (default: green)")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--device", "-d", type=click.Choice(["cuda", "cpu"]), help="Device for inference")
@click.option("--continue-on-error/--abort-on-error", default=True, help="Continue batch if one fails")
@click.option("--parallel", "-j", type=int, default=1, help="Parallel workers for batch")
def matte(
    video_input: str,
    model: str,
    background: str,
    output_dir: Optional[str],
    device: Optional[str],
    continue_on_error: bool,
    parallel: int,
) -> None:
    """Remove background from video (matting).

    Supports batch input: single file, directory, glob pattern, or @filelist.txt

    Examples:

        vidtool matte video.mp4

        vidtool matte ./videos/ -m modnet --parallel 2

        vidtool matte "*.mp4" -b "0,0,255"
    """
    from video_toolkit.utils import BatchInput, run_batch

    try:
        config = Config.from_env()
        if device:
            config.device = device
        if output_dir:
            config.output_dir = Path(output_dir)

        config.ensure_dirs()
        _show_warnings(config)

        bg_color = tuple(int(x) for x in background.split(","))
        if len(bg_color) != 3:
            raise ToolkitError("Background must be R,G,B format (e.g., '0,255,0')")

        batch = BatchInput(video_input)
        videos = batch.resolve()

        if not videos:
            click.secho("No videos found matching input", fg="yellow")
            return

        click.echo(f"\nVideo AI Toolkit - Matte")
        click.echo("=" * 40)
        click.echo(f"Model: {model}")
        click.echo(f"Background: RGB{bg_color}")
        click.echo(f"Videos: {len(videos)}")
        click.echo("")

        if model == "rvm":
            from video_toolkit.matting import RVMatter
            matter = RVMatter(config)
            process_fn = lambda vp: matter.remove_background(str(vp), background_color=bg_color)
        else:  # modnet
            from video_toolkit.matting import MODNetMatter
            matter = MODNetMatter(config)
            process_fn = lambda vp: matter.remove_background(str(vp), background_color=bg_color)

        if len(videos) == 1:
            result = process_fn(videos[0])
            click.echo("")
            click.secho("Background removal complete!", fg="green")
        else:
            batch_result = run_batch(batch, process_fn, "matte", parallel, continue_on_error)
            click.echo("")
            click.secho(f"Batch complete: {batch_result.success_count}/{batch_result.total_count}", fg="green")
            if batch_result.errors:
                click.secho(f"Failed: {batch_result.error_count}", fg="red")
            batch_result.save_summary(config.output_dir)

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
# UPSCALE Command
# ============================================================================
@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--scale", "-s", type=click.Choice(["2", "4"]), default="4", help="Upscale factor")
@click.option("--model", "-m", type=click.Choice(["realesrgan", "video2x"]), default="realesrgan", help="Upscaling model")
@click.option("--preset", "-p", type=click.Choice(["general", "anime", "fast"]), default="general", help="Model preset (realesrgan only)")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--device", "-d", type=click.Choice(["cuda", "cpu"]), help="Device for inference")
def upscale(
    video_path: str,
    scale: str,
    model: str,
    preset: str,
    output_dir: Optional[str],
    device: Optional[str],
) -> None:
    """Upscale video resolution.

    Uses AI models to increase video resolution while preserving detail.
    Supports both general content and anime-specific models.

    Examples:

        vidtool upscale video.mp4

        vidtool upscale video.mp4 -s 2 -p anime

        vidtool upscale video.mp4 -m video2x
    """
    try:
        config = Config.from_env()
        if device:
            config.device = device
        if output_dir:
            config.output_dir = Path(output_dir)

        config.ensure_dirs()
        _show_warnings(config)

        click.echo(f"\nVideo AI Toolkit - Upscale")
        click.echo("=" * 40)
        click.echo(f"Model: {model} ({preset})")
        click.echo(f"Scale: {scale}x")
        click.echo(f"Input: {video_path}")
        click.echo("")

        if model == "realesrgan":
            from video_toolkit.upscale import RealESRGANUpscaler
            upscaler = RealESRGANUpscaler(config)
            result = upscaler.upscale_video(video_path, int(scale), preset)
        else:  # video2x
            from video_toolkit.upscale import Video2XUpscaler
            upscaler = Video2XUpscaler(config)
            result = upscaler.upscale_video(video_path, int(scale))

        click.echo("")
        click.secho("Upscaling complete!", fg="green")
        click.echo(f"Output in {config.output_dir}")

    except ToolkitError as e:
        click.secho(f"Error: {e}", fg="red")
        raise SystemExit(1)


# ============================================================================
# INTERPOLATE Command
# ============================================================================
@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--model", "-m", type=click.Choice(["rife", "film"]), default="rife", help="Interpolation model")
@click.option("--multiplier", "-x", type=click.Choice(["2", "4", "8"]), default="2", help="Frame multiplier")
@click.option("--fps", "-f", type=float, help="Target FPS (overrides multiplier)")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--device", "-d", type=click.Choice(["cuda", "cpu"]), help="Device for inference")
def interpolate(
    video_path: str,
    model: str,
    multiplier: str,
    fps: Optional[float],
    output_dir: Optional[str],
    device: Optional[str],
) -> None:
    """Interpolate video frames for slow motion.

    Generates intermediate frames to increase frame rate or create
    smooth slow motion effects.

    Examples:

        vidtool interpolate video.mp4

        vidtool interpolate video.mp4 -x 4

        vidtool interpolate video.mp4 --fps 60 -m film
    """
    try:
        config = Config.from_env()
        if device:
            config.device = device
        if output_dir:
            config.output_dir = Path(output_dir)

        config.ensure_dirs()
        _show_warnings(config)

        click.echo(f"\nVideo AI Toolkit - Interpolate")
        click.echo("=" * 40)
        click.echo(f"Model: {model}")
        if fps:
            click.echo(f"Target FPS: {fps}")
        else:
            click.echo(f"Multiplier: {multiplier}x")
        click.echo(f"Input: {video_path}")
        click.echo("")

        if model == "rife":
            from video_toolkit.interpolate import RIFEInterpolator
            interpolator = RIFEInterpolator(config)
            result = interpolator.interpolate_video(video_path, int(multiplier), fps)
        else:  # film
            from video_toolkit.interpolate import FILMInterpolator
            interpolator = FILMInterpolator(config)
            result = interpolator.interpolate_video(video_path, int(multiplier), fps)

        click.echo("")
        click.secho("Interpolation complete!", fg="green")
        click.echo(f"Output in {config.output_dir}")

    except ToolkitError as e:
        click.secho(f"Error: {e}", fg="red")
        raise SystemExit(1)


# ============================================================================
# FACE Command
# ============================================================================
@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--model", "-m", type=click.Choice(["gfpgan", "codeformer"]), default="gfpgan", help="Face restoration model")
@click.option("--fidelity", "-f", type=float, default=0.5, help="Fidelity weight (codeformer only, 0.0-1.0)")
@click.option("--upscale", "-s", type=click.Choice(["1", "2", "4"]), default="2", help="Upscale factor")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--device", "-d", type=click.Choice(["cuda", "cpu"]), help="Device for inference")
def face(
    video_path: str,
    model: str,
    fidelity: float,
    upscale: str,
    output_dir: Optional[str],
    device: Optional[str],
) -> None:
    """Restore faces in video.

    Enhances and restores degraded faces using AI models.
    Useful for old footage or low-quality video restoration.

    Examples:

        vidtool face video.mp4

        vidtool face video.mp4 -m codeformer -f 0.7

        vidtool face video.mp4 -s 4
    """
    try:
        config = Config.from_env()
        if device:
            config.device = device
        if output_dir:
            config.output_dir = Path(output_dir)

        config.ensure_dirs()
        _show_warnings(config)

        click.echo(f"\nVideo AI Toolkit - Face Restoration")
        click.echo("=" * 40)
        click.echo(f"Model: {model}")
        if model == "codeformer":
            click.echo(f"Fidelity: {fidelity}")
        click.echo(f"Upscale: {upscale}x")
        click.echo(f"Input: {video_path}")
        click.echo("")

        if model == "gfpgan":
            from video_toolkit.face import GFPGANRestorer
            restorer = GFPGANRestorer(config)
            result = restorer.restore_video(video_path, int(upscale))
        else:  # codeformer
            from video_toolkit.face import CodeFormerRestorer
            restorer = CodeFormerRestorer(config)
            result = restorer.restore_video(video_path, fidelity, int(upscale))

        click.echo("")
        click.secho("Face restoration complete!", fg="green")
        click.echo(f"Output in {config.output_dir}")

    except ToolkitError as e:
        click.secho(f"Error: {e}", fg="red")
        raise SystemExit(1)


# ============================================================================
# FLOW Command
# ============================================================================
@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--model", "-m", type=click.Choice(["raft", "unimatch"]), default="raft", help="Optical flow model")
@click.option("--variant", "-v", type=click.Choice(["small", "standard"]), default="standard", help="Model variant (raft only)")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--device", "-d", type=click.Choice(["cuda", "cpu"]), help="Device for inference")
def flow(
    video_path: str,
    model: str,
    variant: str,
    output_dir: Optional[str],
    device: Optional[str],
) -> None:
    """Estimate optical flow in video.

    Computes dense motion vectors between consecutive frames.
    Outputs colorized flow visualization.

    Examples:

        vidtool flow video.mp4

        vidtool flow video.mp4 -m raft -v small

        vidtool flow video.mp4 -m unimatch
    """
    try:
        config = Config.from_env()
        if device:
            config.device = device
        if output_dir:
            config.output_dir = Path(output_dir)

        config.ensure_dirs()
        _show_warnings(config)

        click.echo(f"\nVideo AI Toolkit - Optical Flow")
        click.echo("=" * 40)
        click.echo(f"Model: {model}")
        if model == "raft":
            click.echo(f"Variant: {variant}")
        click.echo(f"Input: {video_path}")
        click.echo("")

        if model == "raft":
            from video_toolkit.flow import RAFTEstimator
            estimator = RAFTEstimator(config)
            result = estimator.estimate_flow(video_path, variant)
        else:  # unimatch
            from video_toolkit.flow import UniMatchEstimator
            estimator = UniMatchEstimator(config)
            result = estimator.estimate_flow(video_path)

        click.echo("")
        click.secho("Flow estimation complete!", fg="green")
        click.echo(f"Output in {config.output_dir}")

    except ToolkitError as e:
        click.secho(f"Error: {e}", fg="red")
        raise SystemExit(1)


# ============================================================================
# STABILIZE Command
# ============================================================================
@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--smoothing", "-s", type=float, default=0.8, help="Smoothing factor (0.0-1.0)")
@click.option("--crop", "-c", type=float, default=0.9, help="Crop ratio to hide borders (0.0-1.0)")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--device", "-d", type=click.Choice(["cuda", "cpu"]), help="Device for inference")
def stabilize(
    video_path: str,
    smoothing: float,
    crop: float,
    output_dir: Optional[str],
    device: Optional[str],
) -> None:
    """Stabilize shaky video.

    Uses optical flow and trajectory smoothing to remove camera shake.

    Examples:

        vidtool stabilize shaky.mp4

        vidtool stabilize shaky.mp4 -s 0.9 -c 0.85

        vidtool stabilize shaky.mp4 -o ./stabilized
    """
    try:
        config = Config.from_env()
        if device:
            config.device = device
        if output_dir:
            config.output_dir = Path(output_dir)

        config.ensure_dirs()
        _show_warnings(config)

        click.echo(f"\nVideo AI Toolkit - Stabilize")
        click.echo("=" * 40)
        click.echo(f"Smoothing: {smoothing}")
        click.echo(f"Crop ratio: {crop}")
        click.echo(f"Input: {video_path}")
        click.echo("")

        from video_toolkit.stabilize import DeepStabilizer
        stabilizer = DeepStabilizer(config)
        result = stabilizer.stabilize_video(video_path, smoothing, crop)

        click.echo("")
        click.secho("Stabilization complete!", fg="green")
        click.echo(f"Output in {config.output_dir}")

    except ToolkitError as e:
        click.secho(f"Error: {e}", fg="red")
        raise SystemExit(1)


# ============================================================================
# DENOISE Command
# ============================================================================
@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--model", "-m", type=click.Choice(["fastdvdnet", "videnn"]), default="fastdvdnet", help="Denoising model")
@click.option("--sigma", "-s", type=float, default=25.0, help="Noise level (fastdvdnet: 0-50)")
@click.option("--strength", type=float, default=1.0, help="Denoising strength (videnn: 0.0-2.0)")
@click.option("--low-light", is_flag=True, help="Enable low-light enhancement (videnn only)")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--device", "-d", type=click.Choice(["cuda", "cpu"]), help="Device for inference")
def denoise(
    video_path: str,
    model: str,
    sigma: float,
    strength: float,
    low_light: bool,
    output_dir: Optional[str],
    device: Optional[str],
) -> None:
    """Remove noise from video.

    Uses temporal and spatial filtering to reduce video noise.

    Examples:

        vidtool denoise noisy.mp4

        vidtool denoise noisy.mp4 -m fastdvdnet -s 30

        vidtool denoise dark.mp4 -m videnn --low-light
    """
    try:
        config = Config.from_env()
        if device:
            config.device = device
        if output_dir:
            config.output_dir = Path(output_dir)

        config.ensure_dirs()
        _show_warnings(config)

        click.echo(f"\nVideo AI Toolkit - Denoise")
        click.echo("=" * 40)
        click.echo(f"Model: {model}")
        if model == "fastdvdnet":
            click.echo(f"Noise sigma: {sigma}")
        else:
            click.echo(f"Strength: {strength}")
            if low_light:
                click.echo("Low-light enhancement: enabled")
        click.echo(f"Input: {video_path}")
        click.echo("")

        if model == "fastdvdnet":
            from video_toolkit.denoise import FastDVDnetDenoiser
            denoiser = FastDVDnetDenoiser(config)
            result = denoiser.denoise_video(video_path, sigma)
        else:  # videnn
            from video_toolkit.denoise import ViDeNNDenoiser
            denoiser = ViDeNNDenoiser(config)
            result = denoiser.denoise_video(video_path, strength, low_light)

        click.echo("")
        click.secho("Denoising complete!", fg="green")
        click.echo(f"Output in {config.output_dir}")

    except ToolkitError as e:
        click.secho(f"Error: {e}", fg="red")
        raise SystemExit(1)


# ============================================================================
# COLORIZE Command
# ============================================================================
@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--model", "-m", type=click.Choice(["deoldify"]), default="deoldify", help="Colorization model")
@click.option("--render-factor", "-r", type=int, default=21, help="Quality factor (10-40, higher = better)")
@click.option("--artistic", is_flag=True, help="Use artistic model for more vibrant colors")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--device", "-d", type=click.Choice(["cuda", "cpu"]), help="Device for inference")
def colorize(
    video_path: str,
    model: str,
    render_factor: int,
    artistic: bool,
    output_dir: Optional[str],
    device: Optional[str],
) -> None:
    """Colorize black and white video.

    Uses AI to add realistic colors to grayscale footage.

    Examples:

        vidtool colorize bw_footage.mp4

        vidtool colorize old_film.mp4 -r 35 --artistic

        vidtool colorize archive.mp4 -o ./colorized
    """
    try:
        config = Config.from_env()
        if device:
            config.device = device
        if output_dir:
            config.output_dir = Path(output_dir)

        config.ensure_dirs()
        _show_warnings(config)

        click.echo(f"\nVideo AI Toolkit - Colorize")
        click.echo("=" * 40)
        click.echo(f"Model: {model}")
        click.echo(f"Render factor: {render_factor}")
        if artistic:
            click.echo("Mode: artistic")
        click.echo(f"Input: {video_path}")
        click.echo("")

        from video_toolkit.colorize import DeOldifyColorizer
        colorizer = DeOldifyColorizer(config)
        result = colorizer.colorize_video(video_path, render_factor, artistic)

        click.echo("")
        click.secho("Colorization complete!", fg="green")
        click.echo(f"Output in {config.output_dir}")

    except ToolkitError as e:
        click.secho(f"Error: {e}", fg="red")
        raise SystemExit(1)


# ============================================================================
# GENERATE Command
# ============================================================================
@cli.command()
@click.option("--prompt", "-p", required=True, help="Text description of video to generate")
@click.option("--image", "-i", type=click.Path(exists=True), help="Source image for image-to-video")
@click.option("--model", "-m", type=click.Choice(["wan", "cogvideo"]), default="wan", help="Generation model")
@click.option("--frames", "-f", type=int, default=49, help="Number of frames to generate")
@click.option("--seed", "-s", type=int, help="Random seed for reproducibility")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--device", "-d", type=click.Choice(["cuda", "cpu"]), help="Device for inference")
def generate(
    prompt: str,
    image: Optional[str],
    model: str,
    frames: int,
    seed: Optional[int],
    output_dir: Optional[str],
    device: Optional[str],
) -> None:
    """Generate video from text or image.

    Uses AI to create video from text descriptions or animate images.

    Examples:

        vidtool generate -p "a cat walking on the beach"

        vidtool generate -p "ocean waves" -m cogvideo -f 81

        vidtool generate -p "person dancing" -i photo.jpg
    """
    try:
        config = Config.from_env()
        if device:
            config.device = device
        if output_dir:
            config.output_dir = Path(output_dir)

        config.ensure_dirs()
        _show_warnings(config)

        click.echo(f"\nVideo AI Toolkit - Generate")
        click.echo("=" * 40)
        click.echo(f"Model: {model}")
        click.echo(f"Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Prompt: {prompt}")
        if image:
            click.echo(f"Image: {image}")
        click.echo(f"Frames: {frames}")
        if seed:
            click.echo(f"Seed: {seed}")
        click.echo("")

        if model == "wan":
            from video_toolkit.generate import WanGenerator
            generator = WanGenerator(config)
            if image:
                result = generator.generate_from_image(image, prompt, frames, seed=seed)
            else:
                result = generator.generate_from_text(prompt, num_frames=frames, seed=seed)
        else:  # cogvideo
            from video_toolkit.generate import CogVideoGenerator
            generator = CogVideoGenerator(config)
            if image:
                result = generator.generate_from_image(image, prompt, frames, seed=seed)
            else:
                result = generator.generate_from_text(prompt, num_frames=frames, seed=seed)

        click.echo("")
        click.secho("Generation complete!", fg="green")
        click.echo(f"Output in {config.output_dir}")

    except ToolkitError as e:
        click.secho(f"Error: {e}", fg="red")
        raise SystemExit(1)


# ============================================================================
# STYLE Command
# ============================================================================
@cli.command()
@click.argument("video_input")
@click.option("--style", "-s", required=True, help="Style image path or pre-trained style name")
@click.option("--model", "-m", type=click.Choice(["adain", "fast-artistic"]), default="adain", help="Style transfer model")
@click.option("--alpha", "-a", type=float, default=1.0, help="Style strength 0.0-1.0 (adain only)")
@click.option("--preserve-color", is_flag=True, help="Keep original colors (adain only)")
@click.option("--temporal-weight", "-t", type=float, default=0.5, help="Temporal consistency 0.0-1.0 (fast-artistic only)")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--device", "-d", type=click.Choice(["cuda", "cpu"]), help="Device for inference")
@click.option("--continue-on-error/--abort-on-error", default=True, help="Continue batch if one fails")
@click.option("--parallel", "-j", type=int, default=1, help="Parallel workers for batch")
def style(
    video_input: str,
    style: str,
    model: str,
    alpha: float,
    preserve_color: bool,
    temporal_weight: float,
    output_dir: Optional[str],
    device: Optional[str],
    continue_on_error: bool,
    parallel: int,
) -> None:
    """Apply artistic style transfer to video.

    Supports batch input: single file, directory, glob pattern, or @filelist.txt

    Examples:

        vidtool style video.mp4 --style painting.jpg

        vidtool style video.mp4 --style la_muse -m fast-artistic

        vidtool style ./videos/ --style art.jpg --alpha 0.7
    """
    from video_toolkit.utils import BatchInput, run_batch

    try:
        config = Config.from_env()
        if device:
            config.device = device
        if output_dir:
            config.output_dir = Path(output_dir)

        config.ensure_dirs()
        _show_warnings(config)

        # Resolve batch input
        batch = BatchInput(video_input)
        videos = batch.resolve()

        if not videos:
            click.secho("No videos found matching input", fg="yellow")
            return

        click.echo(f"\nVideo AI Toolkit - Style Transfer")
        click.echo("=" * 40)
        click.echo(f"Model: {model}")
        click.echo(f"Style: {style}")
        if model == "adain":
            click.echo(f"Alpha: {alpha}")
            if preserve_color:
                click.echo("Preserve color: enabled")
        else:
            click.echo(f"Temporal weight: {temporal_weight}")
        click.echo(f"Videos: {len(videos)}")
        click.echo("")

        style_path = Path(style)

        if model == "adain":
            from video_toolkit.style import AdaINStyler

            if not style_path.exists():
                raise ToolkitError(f"Style image not found: {style}")

            styler = AdaINStyler(config)

            def process_one(vp: Path):
                return styler.stylize_video(vp, style_path, alpha, preserve_color)

        else:  # fast-artistic
            from video_toolkit.style import FastArtisticStyler

            styler = FastArtisticStyler(config)

            if style not in FastArtisticStyler.AVAILABLE_STYLES:
                available = ", ".join(FastArtisticStyler.AVAILABLE_STYLES)
                raise ToolkitError(f"Unknown style: {style}. Available: {available}")

            def process_one(vp: Path):
                return styler.stylize_video(vp, style, temporal_weight)

        if len(videos) == 1:
            # Single video
            result = process_one(videos[0])
            click.echo("")
            click.secho("Style transfer complete!", fg="green")
            click.echo(f"Output in {config.output_dir}")
        else:
            # Batch mode
            batch_result = run_batch(
                batch,
                process_one,
                "style",
                parallel=parallel,
                continue_on_error=continue_on_error,
            )

            click.echo("")
            click.secho(f"Batch complete: {batch_result.success_count}/{batch_result.total_count}", fg="green")

            if batch_result.errors:
                click.secho(f"Failed: {batch_result.error_count}", fg="red")
                for path, err in batch_result.errors[:5]:
                    click.echo(f"  - {path.name}: {err}")

            batch_result.save_summary(config.output_dir)

    except ToolkitError as e:
        click.secho(f"Error: {e}", fg="red")
        raise SystemExit(1)


# ============================================================================
# PIPELINE Command Group
# ============================================================================
@cli.group()
def pipeline() -> None:
    """Pipeline processing - chain multiple processors.

    Build and run multi-step processing pipelines:

        vidtool pipeline run video.mp4 --steps "denoise,upscale:scale=4"

        vidtool pipeline random video.mp4 --min-steps 2 --max-steps 4

        vidtool pipeline list --category enhancement
    """
    pass


@pipeline.command("run")
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--config", "-c", type=click.Path(exists=True), help="Pipeline config file (YAML/JSON)")
@click.option("--steps", "-s", help="Inline steps: 'denoise,upscale:scale=4,style:style=art.jpg'")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--device", "-d", type=click.Choice(["cuda", "cpu"]), help="Device for inference")
def pipeline_run(
    video_path: str,
    config: Optional[str],
    steps: Optional[str],
    output_dir: Optional[str],
    device: Optional[str],
) -> None:
    """Run a processing pipeline on video.

    Use --config to load from file or --steps for inline specification.

    Examples:

        vidtool pipeline run video.mp4 --config enhance.yaml

        vidtool pipeline run video.mp4 --steps "denoise,upscale:scale=4"

        vidtool pipeline run video.mp4 -s "depth,style:style=art.jpg"
    """
    from video_toolkit.pipeline import Pipeline

    try:
        cfg = Config.from_env()
        if device:
            cfg.device = device
        if output_dir:
            cfg.output_dir = Path(output_dir)
        cfg.ensure_dirs()

        if config:
            pipe = Pipeline.load(config, cfg)
        elif steps:
            pipe = Pipeline.from_steps_string(steps, name="inline", config=cfg)
        else:
            raise ToolkitError("Specify --config or --steps")

        click.echo(f"\nVideo AI Toolkit - Pipeline")
        click.echo("=" * 40)
        result = pipe.run(Path(video_path), cfg.output_dir)

        if result.success:
            click.secho("\nPipeline complete!", fg="green")
            if result.final_output:
                click.echo(f"Final output: {result.final_output}")
        else:
            click.secho(f"\nPipeline failed at step {result.completed_count + 1}", fg="red")
            raise SystemExit(1)

    except ToolkitError as e:
        click.secho(f"Error: {e}", fg="red")
        raise SystemExit(1)


@pipeline.command("random")
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--min-steps", type=int, default=2, help="Minimum number of steps")
@click.option("--max-steps", type=int, default=4, help="Maximum number of steps")
@click.option("--categories", help="Comma-separated categories (enhancement,creative,...)")
@click.option("--exclude", help="Comma-separated processor IDs to exclude")
@click.option("--seed", type=int, help="Random seed for reproducibility")
@click.option("--preview", is_flag=True, help="Preview pipeline without running")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--device", "-d", type=click.Choice(["cuda", "cpu"]), help="Device for inference")
def pipeline_random(
    video_path: str,
    min_steps: int,
    max_steps: int,
    categories: Optional[str],
    exclude: Optional[str],
    seed: Optional[int],
    preview: bool,
    output_dir: Optional[str],
    device: Optional[str],
) -> None:
    """Generate and run a random pipeline.

    Create experimental processing chains with random processor selection.

    Examples:

        vidtool pipeline random video.mp4 --seed 42

        vidtool pipeline random video.mp4 --categories "enhancement,creative"

        vidtool pipeline random video.mp4 --min-steps 2 --max-steps 5 --exclude "generate"

        vidtool pipeline random video.mp4 --preview
    """
    from video_toolkit.pipeline import RandomPipeline

    try:
        cfg = Config.from_env()
        if device:
            cfg.device = device
        if output_dir:
            cfg.output_dir = Path(output_dir)
        cfg.ensure_dirs()

        cat_list = [c.strip() for c in categories.split(",")] if categories else None
        exc_list = [e.strip() for e in exclude.split(",")] if exclude else None

        generator = RandomPipeline(
            min_steps=min_steps,
            max_steps=max_steps,
            categories=cat_list,
            exclude=exc_list,
            config=cfg,
        )

        if preview:
            click.echo(generator.preview(seed))
            return

        click.echo(f"\nVideo AI Toolkit - Random Pipeline")
        click.echo("=" * 40)
        result = generator.run_random(Path(video_path), seed=seed, output_dir=cfg.output_dir)

        if result.success:
            click.secho("\nRandom pipeline complete!", fg="green")
            if result.final_output:
                click.echo(f"Final output: {result.final_output}")
        else:
            click.secho(f"\nPipeline failed at step {result.completed_count + 1}", fg="red")
            raise SystemExit(1)

    except ToolkitError as e:
        click.secho(f"Error: {e}", fg="red")
        raise SystemExit(1)


@pipeline.command("list")
@click.option("--category", "-c", help="Filter by category")
def pipeline_list(category: Optional[str]) -> None:
    """List available processors for pipelines.

    Examples:

        vidtool pipeline list

        vidtool pipeline list --category enhancement
    """
    from video_toolkit.pipeline import ProcessorRegistry

    click.echo("\n" + ProcessorRegistry.format_list(category))


@pipeline.command("create")
@click.argument("output_file", type=click.Path())
@click.option("--steps", "-s", required=True, help="Steps: 'denoise,upscale:scale=4,style:style=art.jpg'")
@click.option("--name", "-n", default="my_pipeline", help="Pipeline name")
@click.option("--description", "-d", default="", help="Pipeline description")
def pipeline_create(
    output_file: str,
    steps: str,
    name: str,
    description: str,
) -> None:
    """Create a pipeline config file from steps.

    Examples:

        vidtool pipeline create enhance.yaml --steps "denoise,upscale:scale=4"

        vidtool pipeline create my.json -s "depth,style:style=art.jpg" -n "my_pipe"
    """
    from video_toolkit.pipeline import Pipeline

    try:
        pipe = Pipeline.from_steps_string(steps, name=name)
        pipe.description = description
        pipe.save(Path(output_file))

    except Exception as e:
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
