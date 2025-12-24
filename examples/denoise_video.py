#!/usr/bin/env python3
"""Example: Remove noise from video."""

from pathlib import Path
from video_toolkit.config import Config
from video_toolkit.denoise import FastDVDnetDenoiser, ViDeNNDenoiser


def main():
    config = Config.from_env()
    config.ensure_dirs()

    # Method 1: FastDVDnet - fast, no motion compensation needed
    denoiser = FastDVDnetDenoiser(config)
    result = denoiser.denoise_video(
        "noisy_video.mp4",
        noise_sigma=25.0,  # Estimated noise level (0-50)
        temporal_window=5,  # Number of frames for temporal averaging
    )
    print(f"FastDVDnet output: {result.outputs[0].path}")

    # Method 2: ViDeNN - good for low-light footage
    denoiser2 = ViDeNNDenoiser(config)
    result2 = denoiser2.denoise_video(
        "dark_video.mp4",
        strength=1.0,    # Denoising strength (0.0-2.0)
        low_light=True,  # Enable CLAHE enhancement
    )
    print(f"ViDeNN output: {result2.outputs[0].path}")


if __name__ == "__main__":
    main()
