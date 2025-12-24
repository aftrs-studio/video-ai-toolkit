#!/usr/bin/env python3
"""Example: Generate video from text or image."""

from pathlib import Path
from video_toolkit.config import Config
from video_toolkit.generate import WanGenerator, CogVideoGenerator


def main():
    config = Config.from_env()
    config.ensure_dirs()

    # Method 1: Wan 2.1 - Alibaba's text-to-video
    wan = WanGenerator(config)

    # Text to video
    result = wan.generate_from_text(
        prompt="A golden retriever playing on a sunny beach",
        negative_prompt="blurry, low quality",
        num_frames=81,  # ~3 seconds at 24fps
        seed=42,
    )
    print(f"Wan T2V output: {result.outputs[0].path}")

    # Image to video
    result2 = wan.generate_from_image(
        image_path="photo.jpg",
        prompt="gentle movement, cinematic",
        num_frames=81,
        seed=42,
    )
    print(f"Wan I2V output: {result2.outputs[0].path}")

    # Method 2: CogVideoX - Zhipu AI's model
    cogvideo = CogVideoGenerator(config)

    result3 = cogvideo.generate_from_text(
        prompt="A cat walking through a garden",
        num_frames=49,  # ~6 seconds at 8fps
        variant="5b",   # 2b or 5b
        seed=42,
    )
    print(f"CogVideoX output: {result3.outputs[0].path}")


if __name__ == "__main__":
    main()
