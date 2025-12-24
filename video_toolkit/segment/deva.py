"""DEVA - Decoupled Video Segmentation."""

from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from video_toolkit.config import Config
from video_toolkit.utils import (
    ModelNotFoundError,
    ToolkitError,
    get_output_filename,
    parse_concepts,
)
from video_toolkit.video_io import ProcessingResult, VideoReader, VideoWriter


class DEVASegmenter:
    """Video segmentation using DEVA (Tracking Anything with Decoupled Video Segmentation).

    DEVA uses decoupled image-level segmentation and temporal propagation
    for efficient video object segmentation.

    GitHub: https://github.com/hkchengrex/Tracking-Anything-with-DEVA
    """

    MODEL_ID = "deva"
    MODEL_NAME = "DEVA"

    def __init__(self, config: Optional[Config] = None):
        """Initialize DEVA segmenter.

        Args:
            config: Configuration object (loads from env if not provided)
        """
        self.config = config or Config.from_env()
        self._model = None
        self._sam = None
        self._grounding_dino = None

    def _load_model(self) -> None:
        """Load DEVA model and components."""
        if self._model is not None:
            return

        try:
            from deva.inference.inference_core import DEVAInferenceCore
            from deva.inference.eval_args import get_model_and_config
        except ImportError:
            raise ToolkitError(
                "DEVA not installed. Run:\n"
                "  pip install git+https://github.com/hkchengrex/Tracking-Anything-with-DEVA.git\n"
                "Or use: ./scripts/install_segment.sh"
            )

        print(f"Loading {self.MODEL_NAME} on {self.config.device}...")

        try:
            model_dir = self.config.get_model_dir(self.MODEL_ID)

            # Load DEVA core
            network, config, _ = get_model_and_config(
                model_path=str(model_dir / "DEVA-propagation.pth")
            )
            self._model = DEVAInferenceCore(network, config)
            self._model.to(self.config.device)

        except Exception as e:
            raise ModelNotFoundError(f"Failed to load {self.MODEL_NAME}: {e}")

        print(f"{self.MODEL_NAME} loaded successfully")

    def segment_video(
        self,
        video_path: str | Path,
        concepts: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> ProcessingResult:
        """Segment objects from video.

        Args:
            video_path: Path to input video
            concepts: Optional comma-separated text prompts
            output_dir: Output directory

        Returns:
            ProcessingResult with paths to segmented videos
        """
        video_path = Path(video_path)
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        concept_list = parse_concepts(concepts) if concepts else []

        self._load_model()

        result = ProcessingResult(video_path, output_dir, "segment")

        with VideoReader(video_path, self.config.max_frames) as reader:
            info = reader.info
            print(f"Video: {info['width']}x{info['height']} @ {info['fps']:.1f}fps")
            if concept_list:
                print(f"Concepts: {', '.join(concept_list)}")

            frames = reader.read_all()
            print(f"Loaded {len(frames)} frames")

            # Process with DEVA
            print("Processing with DEVA...")
            video_segments = {}

            # Initialize DEVA state
            self._model.clear_memory()

            for frame_idx, frame in enumerate(tqdm(frames, desc="Segmenting")):
                # Get segmentation for this frame
                if frame_idx == 0:
                    # First frame: run detection
                    masks, labels = self._get_initial_masks(frame, concept_list)
                    for obj_id, (mask, label) in enumerate(zip(masks, labels)):
                        video_segments[obj_id + 1] = {
                            "masks": [mask],
                            "label": label,
                        }
                        self._model.add_mask(mask, obj_id + 1)
                else:
                    # Subsequent frames: propagate
                    output = self._model.step(frame)
                    for obj_id, mask in output.items():
                        if obj_id in video_segments:
                            video_segments[obj_id]["masks"].append(mask)

            # Write output videos
            print(f"Writing {len(video_segments)} segmented videos...")
            for obj_id, data in video_segments.items():
                label = data.get("label", "object")
                masks = data["masks"]

                output_path = get_output_filename(
                    video_path, "segment", label, output_dir, instance_id=obj_id
                )

                with VideoWriter(
                    output_path, info["width"], info["height"], info["fps"]
                ) as writer:
                    for frame_idx, mask in enumerate(masks):
                        if frame_idx < len(frames):
                            writer.write_frame(frames[frame_idx], mask)

                result.add_output(
                    output_path, self.MODEL_ID, len(masks),
                    concept=label, object_id=obj_id
                )
                print(f"  Created: {output_path.name}")

        result.save_metadata()
        result.save_summary()

        return result

    def _get_initial_masks(
        self,
        frame: np.ndarray,
        concepts: list[str],
    ) -> tuple[list[np.ndarray], list[str]]:
        """Get initial masks for first frame.

        Args:
            frame: First frame as RGB array
            concepts: List of concepts to detect

        Returns:
            Tuple of (masks list, labels list)
        """
        # This would use Grounding DINO + SAM for initial detection
        # Simplified placeholder for the wrapper
        masks = []
        labels = []

        if concepts:
            # Use text-prompted detection
            # In full implementation, this would call Grounding DINO
            pass
        else:
            # Auto-detect all objects
            # In full implementation, this would run automatic segmentation
            pass

        return masks, labels

    def get_info(self) -> dict:
        """Get model information."""
        import torch

        return {
            "model": self.MODEL_NAME,
            "model_id": self.MODEL_ID,
            "paper": "ICCV 2023",
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": self.config.device,
        }
