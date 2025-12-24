"""SAM 2 (Segment Anything Model 2) video segmentation."""

from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from video_toolkit.config import Config
from video_toolkit.utils import ModelNotFoundError, ToolkitError, get_output_filename
from video_toolkit.video_io import ProcessingResult, VideoReader, VideoWriter


class SAM2Segmenter:
    """Video segmentation using Meta's SAM 2.

    SAM 2 provides real-time video segmentation at 44 FPS with
    flexible prompting through points, bounding boxes, or masks.

    GitHub: https://github.com/facebookresearch/sam2
    """

    MODEL_ID = "sam2"
    MODEL_NAME = "SAM 2"

    def __init__(self, config: Optional[Config] = None):
        """Initialize SAM 2 segmenter.

        Args:
            config: Configuration object (loads from env if not provided)
        """
        self.config = config or Config.from_env()
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        """Load SAM 2 model and processor."""
        if self._model is not None:
            return

        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            raise ToolkitError(
                "SAM 2 not installed. Run: pip install segment-anything-2\n"
                "Or use: ./scripts/install_segment.sh"
            )

        print(f"Loading {self.MODEL_NAME} on {self.config.device}...")

        try:
            model_dir = self.config.get_model_dir(self.MODEL_ID)
            self._model = build_sam2_video_predictor(
                config_file="sam2_hiera_l.yaml",
                ckpt_path=str(model_dir / "sam2_hiera_large.pt"),
                device=self.config.device,
            )
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load {self.MODEL_NAME}: {e}")

        print(f"{self.MODEL_NAME} loaded successfully")

    def segment_video(
        self,
        video_path: str | Path,
        points: Optional[list[tuple[int, int]]] = None,
        boxes: Optional[list[tuple[int, int, int, int]]] = None,
        output_dir: Optional[Path] = None,
    ) -> ProcessingResult:
        """Segment objects from video using point or box prompts.

        Args:
            video_path: Path to input video
            points: List of (x, y) point prompts
            boxes: List of (x1, y1, x2, y2) box prompts
            output_dir: Output directory

        Returns:
            ProcessingResult with paths to segmented videos
        """
        video_path = Path(video_path)
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        self._load_model()

        result = ProcessingResult(video_path, output_dir, "segment")

        with VideoReader(video_path, self.config.max_frames) as reader:
            info = reader.info
            print(f"Video: {info['width']}x{info['height']} @ {info['fps']:.1f}fps")

            frames = reader.read_all()
            print(f"Loaded {len(frames)} frames")

            # Initialize inference state
            inference_state = self._model.init_state(video_path=str(video_path))

            # Add prompts
            if points:
                for i, point in enumerate(points):
                    self._model.add_new_points(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=i + 1,
                        points=np.array([[point]], dtype=np.float32),
                        labels=np.array([1], dtype=np.int32),
                    )
            elif boxes:
                for i, box in enumerate(boxes):
                    self._model.add_new_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=i + 1,
                        box=np.array(box, dtype=np.float32),
                    )

            # Propagate through video
            print("Propagating masks through video...")
            video_segments = {}

            for out in tqdm(
                self._model.propagate_in_video(inference_state),
                total=len(frames),
                desc="Tracking",
            ):
                frame_idx = out["frame_idx"]
                for obj_id, mask in out["masks"].items():
                    if obj_id not in video_segments:
                        video_segments[obj_id] = []
                    video_segments[obj_id].append(mask)

            # Write output videos
            for obj_id, masks in video_segments.items():
                output_path = get_output_filename(
                    video_path, "segment", f"obj{obj_id}", output_dir
                )

                with VideoWriter(
                    output_path, info["width"], info["height"], info["fps"]
                ) as writer:
                    for frame_idx, mask in enumerate(masks):
                        if frame_idx < len(frames):
                            writer.write_frame(frames[frame_idx], mask)

                result.add_output(output_path, self.MODEL_ID, len(masks), object_id=obj_id)
                print(f"  Created: {output_path.name}")

        result.save_metadata()
        result.save_summary()

        return result

    def get_info(self) -> dict:
        """Get model information.

        Returns:
            Dictionary with model info
        """
        import torch

        return {
            "model": self.MODEL_NAME,
            "model_id": self.MODEL_ID,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": self.config.device,
        }
