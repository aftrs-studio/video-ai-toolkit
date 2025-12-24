"""Grounded SAM 2 - Text-prompted video segmentation and tracking."""

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


class GroundedSAM2Segmenter:
    """Video segmentation using Grounded SAM 2 with text prompts.

    Combines Grounding DINO for text-based detection with SAM 2 for
    precise segmentation and tracking.

    GitHub: https://github.com/IDEA-Research/Grounded-SAM-2
    """

    MODEL_ID = "grounded"
    MODEL_NAME = "Grounded SAM 2"

    def __init__(self, config: Optional[Config] = None):
        """Initialize Grounded SAM 2 segmenter.

        Args:
            config: Configuration object (loads from env if not provided)
        """
        self.config = config or Config.from_env()
        self._sam2_model = None
        self._grounding_model = None

    def _load_model(self) -> None:
        """Load Grounded SAM 2 models."""
        if self._sam2_model is not None:
            return

        try:
            from sam2.build_sam import build_sam2_video_predictor
            from groundingdino.util.inference import Model as GroundingDINOModel
        except ImportError:
            raise ToolkitError(
                "Grounded SAM 2 not installed. Run:\n"
                "  pip install segment-anything-2\n"
                "  pip install groundingdino\n"
                "Or use: ./scripts/install_segment.sh"
            )

        print(f"Loading {self.MODEL_NAME} on {self.config.device}...")

        try:
            model_dir = self.config.get_model_dir(self.MODEL_ID)

            # Load SAM 2
            self._sam2_model = build_sam2_video_predictor(
                config_file="sam2_hiera_l.yaml",
                ckpt_path=str(model_dir / "sam2_hiera_large.pt"),
                device=self.config.device,
            )

            # Load Grounding DINO
            self._grounding_model = GroundingDINOModel(
                model_config_path=str(model_dir / "GroundingDINO_SwinT_OGC.py"),
                model_checkpoint_path=str(model_dir / "groundingdino_swint_ogc.pth"),
                device=self.config.device,
            )
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load {self.MODEL_NAME}: {e}")

        print(f"{self.MODEL_NAME} loaded successfully")

    def segment_video(
        self,
        video_path: str | Path,
        concepts: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        output_dir: Optional[Path] = None,
    ) -> ProcessingResult:
        """Segment objects from video using text prompts.

        Args:
            video_path: Path to input video
            concepts: Comma-separated text prompts (e.g., "person,car,dog")
            box_threshold: Detection confidence threshold
            text_threshold: Text matching threshold
            output_dir: Output directory

        Returns:
            ProcessingResult with paths to segmented videos
        """
        video_path = Path(video_path)
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        concept_list = parse_concepts(concepts)
        if not concept_list:
            raise ToolkitError("No concepts provided. Use -c 'person,car' to specify.")

        self._load_model()

        result = ProcessingResult(video_path, output_dir, "segment")

        with VideoReader(video_path, self.config.max_frames) as reader:
            info = reader.info
            print(f"Video: {info['width']}x{info['height']} @ {info['fps']:.1f}fps")
            print(f"Concepts: {', '.join(concept_list)}")

            frames = reader.read_all()
            print(f"Loaded {len(frames)} frames")

            # Detect objects in first frame using Grounding DINO
            print("Detecting objects with Grounding DINO...")
            first_frame = frames[0]
            prompt = ". ".join(concept_list) + "."

            detections = self._grounding_model.predict_with_caption(
                image=first_frame,
                caption=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            if len(detections.xyxy) == 0:
                print("No objects detected. Try lowering thresholds or different prompts.")
                return result

            print(f"Detected {len(detections.xyxy)} objects")

            # Initialize SAM 2 with detected boxes
            inference_state = self._sam2_model.init_state(video_path=str(video_path))

            for i, (box, label) in enumerate(zip(detections.xyxy, detections.class_id)):
                self._sam2_model.add_new_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=i + 1,
                    box=box.astype(np.float32),
                )

            # Track through video
            print("Tracking objects through video...")
            video_segments = {}
            object_labels = {}

            for i, label_id in enumerate(detections.class_id):
                object_labels[i + 1] = concept_list[label_id] if label_id < len(concept_list) else "object"

            for out in tqdm(
                self._sam2_model.propagate_in_video(inference_state),
                total=len(frames),
                desc="Tracking",
            ):
                frame_idx = out["frame_idx"]
                for obj_id, mask in out["masks"].items():
                    if obj_id not in video_segments:
                        video_segments[obj_id] = []
                    video_segments[obj_id].append(mask)

            # Write output videos
            print(f"Writing {len(video_segments)} segmented videos...")
            for obj_id, masks in video_segments.items():
                label = object_labels.get(obj_id, "object")
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

    def get_info(self) -> dict:
        """Get model information."""
        import torch

        return {
            "model": self.MODEL_NAME,
            "model_id": self.MODEL_ID,
            "components": ["SAM 2", "Grounding DINO"],
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": self.config.device,
        }
