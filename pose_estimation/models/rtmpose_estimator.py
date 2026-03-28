"""RTMPose wrapper implementing BaseEstimator.

Uses MMPose's lower-level inference API directly (not MMPoseInferencer),
so that no person detector is required — bboxes are provided explicitly
or the full image is used as a single bbox.

Usage:
    from pose_estimation.models.rtmpose_estimator import RTMPoseEstimator

    estimator = RTMPoseEstimator(
        config="vendor/mmpose/configs/body_2d_keypoint/rtmpose/..."
        checkpoint="path/to/checkpoint.pth",
        device="mps",  # or "cpu", "cuda"
    )
    results = estimator.predict(image)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from pose_estimation.core.base_estimator import BaseEstimator
from pose_estimation.core.constants import NUM_KEYPOINTS
from pose_estimation.core.types import BBox, Keypoint, PoseResult


def _detect_device() -> str:
    """Auto-detect the best available device."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class RTMPoseEstimator(BaseEstimator):
    """RTMPose pose estimator via MMPose low-level API.

    Unlike MMPoseInferencer, this does NOT require a person detector.
    Bounding boxes must be provided, or the full image is used as bbox.
    """

    def __init__(
        self,
        config: str | Path,
        checkpoint: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.config = str(config)
        self.checkpoint = str(checkpoint) if checkpoint else None
        self.device = device or _detect_device()
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        """Lazily initialize the model."""
        if self._model is not None:
            return

        try:
            from mmpose.apis import init_model
        except ImportError as e:
            raise ImportError(
                "MMPose is not installed. Run setup_env.sh or "
                "uv pip install -e vendor/mmpose"
            ) from e

        device = self.device
        if device == "mps":
            # MMPose may not fully support MPS; fall back to CPU
            warnings.warn(
                "MPS support in MMPose may be unstable. Using CPU.",
                stacklevel=2,
            )
            device = "cpu"

        self._model = init_model(
            self.config,
            self.checkpoint,
            device=device,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a new checkpoint. Reinitializes the model."""
        self.checkpoint = str(path)
        self._model = None

    def predict(
        self,
        image: np.ndarray,
        bboxes: list[BBox] | None = None,
    ) -> list[PoseResult]:
        """Run pose estimation on a single image.

        Args:
            image: BGR uint8 image (H, W, 3).
            bboxes: Optional bounding boxes. If None, uses full image.

        Returns:
            List of PoseResult, one per bbox.
        """
        self._ensure_loaded()

        from mmpose.apis import inference_topdown
        from mmpose.structures import PoseDataSample

        # Prepare bboxes in xyxy format.
        if bboxes is not None:
            np_bboxes = np.array([list(bb.to_xyxy()) for bb in bboxes], dtype=np.float32)
        else:
            h, w = image.shape[:2]
            np_bboxes = np.array([[0, 0, w, h]], dtype=np.float32)

        # Run inference.
        pose_results: list[PoseDataSample] = inference_topdown(
            self._model, image, np_bboxes
        )

        results: list[PoseResult] = []
        for data_sample in pose_results:
            pose = self._parse_data_sample(data_sample)
            if pose is not None:
                results.append(pose)

        return results

    def _parse_data_sample(self, data_sample: Any) -> PoseResult | None:
        """Convert a MMPose PoseDataSample to PoseResult."""
        pred = data_sample.pred_instances

        # Keypoints: shape (N, K, 2) and scores (N, K)
        raw_kps = pred.keypoints  # (N, 17, 2)
        raw_scores = pred.keypoint_scores  # (N, 17)

        if len(raw_kps) == 0:
            return None

        # Take first instance.
        kps = raw_kps[0]  # (17, 2)
        scores = raw_scores[0]  # (17,)

        if len(kps) < NUM_KEYPOINTS:
            return None

        keypoints: list[Keypoint] = []
        for i in range(NUM_KEYPOINTS):
            x, y = float(kps[i][0]), float(kps[i][1])
            conf = float(scores[i])
            vis = 2 if conf > 0.3 else 0
            keypoints.append(Keypoint(x=x, y=y, confidence=conf, visibility=vis))

        # Parse bbox.
        if hasattr(pred, "bboxes") and len(pred.bboxes) > 0:
            x1, y1, x2, y2 = [float(v) for v in pred.bboxes[0]]
            bbox = BBox.from_xyxy(x1, y1, x2, y2)
        else:
            bbox = BBox.from_keypoints(keypoints)

        score = float(pred.bbox_scores[0]) if hasattr(pred, "bbox_scores") and len(pred.bbox_scores) > 0 else 0.0

        return PoseResult(keypoints=keypoints, bbox=bbox, score=score)

    def predict_batch(
        self,
        images: list[np.ndarray],
        bboxes_list: list[list[BBox] | None] | None = None,
    ) -> list[list[PoseResult]]:
        """Run prediction on multiple images."""
        if bboxes_list is None:
            bboxes_list = [None] * len(images)
        return [
            self.predict(img, bbs)
            for img, bbs in zip(images, bboxes_list)
        ]
