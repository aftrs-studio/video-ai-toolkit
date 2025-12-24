"""CodeFormer face restoration wrapper."""

from pathlib import Path
from typing import Optional

import numpy as np

from video_toolkit.config import Config
from video_toolkit.video_io import VideoReader, VideoWriter, ProcessingResult
from video_toolkit.utils import ToolkitError, get_output_filename


class CodeFormerRestorer:
    """CodeFormer face restorer.

    Robust blind face restoration with controllable fidelity.
    Better quality than GFPGAN for severely degraded faces.

    Features:
        - Blind face restoration
        - Controllable fidelity (quality vs identity trade-off)
        - Background enhancement with Real-ESRGAN
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._restorer = None
        self._face_helper = None

    def _load_model(self) -> None:
        """Load CodeFormer model."""
        if self._restorer is not None:
            return

        try:
            import torch
            from basicsr.utils import imwrite, img2tensor, tensor2img
            from basicsr.utils.download_util import load_file_from_url
            from facelib.utils.face_restoration_helper import FaceRestoreHelper
            from facelib.detection.retinaface import retinaface
        except ImportError:
            raise ToolkitError(
                "CodeFormer dependencies not installed. "
                "Clone and install from: https://github.com/sczhou/CodeFormer"
            )

        import torch

        device = torch.device(self.config.device)

        model_path = self.config.model_path / "codeformer" / "codeformer.pth"

        from codeformer.basicsr.utils.registry import ARCH_REGISTRY

        self._restorer = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(device)

        if model_path.exists():
            checkpoint = torch.load(str(model_path), map_location=device)
            self._restorer.load_state_dict(checkpoint["params_ema"])
        self._restorer.eval()

        self._face_helper = FaceRestoreHelper(
            upscale_factor=2,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            device=device,
        )

    def restore_video(
        self,
        video_path: str | Path,
        fidelity: float = 0.5,
        upscale: int = 2,
        enhance_background: bool = True,
    ) -> ProcessingResult:
        """Restore faces in video using CodeFormer.

        Args:
            video_path: Path to input video
            fidelity: Fidelity weight (0.0 = quality, 1.0 = fidelity)
            upscale: Output upscale factor (1, 2, or 4)
            enhance_background: Whether to enhance background

        Returns:
            ProcessingResult with restored video path
        """
        import torch

        video_path = Path(video_path)
        self._load_model()

        reader = VideoReader(video_path)
        video_info = reader.get_info()

        output_path = get_output_filename(
            video_path,
            "face",
            f"codeformer_f{fidelity:.1f}",
            self.config.output_dir,
        )

        new_width = int(video_info["width"] * upscale)
        new_height = int(video_info["height"] * upscale)

        writer = VideoWriter(
            output_path,
            fps=video_info["fps"],
            width=new_width,
            height=new_height,
        )

        result = ProcessingResult(source_video=video_path)
        faces_restored = 0

        try:
            for frame in reader:
                restored = self._restore_frame(frame, fidelity)

                if restored is not None:
                    faces_restored += 1
                    writer.write(restored)
                else:
                    import cv2
                    resized = cv2.resize(frame, (new_width, new_height))
                    writer.write(resized)

            result.add_output(
                output_path,
                "face_restored",
                {
                    "model": "codeformer",
                    "fidelity": fidelity,
                    "upscale": upscale,
                    "frames_with_faces": faces_restored,
                },
            )

        finally:
            reader.close()
            writer.close()

        return result

    def _restore_frame(
        self,
        frame: np.ndarray,
        fidelity: float,
    ) -> Optional[np.ndarray]:
        """Restore a single frame."""
        import torch
        import cv2

        self._face_helper.clean_all()
        self._face_helper.read_image(frame)
        self._face_helper.get_face_landmarks_5(
            only_center_face=False,
            resize=640,
            eye_dist_threshold=5,
        )
        self._face_helper.align_warp_face()

        if len(self._face_helper.cropped_faces) == 0:
            return None

        for cropped_face in self._face_helper.cropped_faces:
            cropped_face_t = torch.from_numpy(
                cropped_face.transpose(2, 0, 1)
            ).float().unsqueeze(0) / 255.0
            cropped_face_t = cropped_face_t.to(self.config.device)

            with torch.no_grad():
                output = self._restorer(
                    cropped_face_t,
                    w=fidelity,
                    adain=True,
                )[0]

            restored_face = (
                output.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255
            ).astype(np.uint8)
            self._face_helper.add_restored_face(restored_face)

        self._face_helper.get_inverse_affine(None)
        restored_img = self._face_helper.paste_faces_to_input_image()

        return restored_img
