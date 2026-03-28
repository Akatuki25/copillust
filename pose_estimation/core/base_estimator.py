"""Abstract base class for pose estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from pose_estimation.core.types import BBox, PoseResult


class BaseEstimator(ABC):
    """Interface that every pose estimation model must implement.

    Subclasses wrap a concrete model (RTMPose, ViTPose, RTMO, etc.)
    and expose a uniform predict API so that inference, evaluation,
    and visualization code can stay model-agnostic.
    """

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        bboxes: list[BBox] | None = None,
    ) -> list[PoseResult]:
        """Run pose estimation on a single image.

        Args:
            image: BGR uint8 image (H, W, 3).
            bboxes: Optional pre-computed bounding boxes.  If ``None``,
                the estimator should either run its own detector or
                treat the full image as a single bbox.

        Returns:
            One ``PoseResult`` per detected person/character.
        """

    @abstractmethod
    def load_checkpoint(self, path: str | Path) -> None:
        """Load model weights from *path*."""

    def predict_batch(
        self,
        images: list[np.ndarray],
        bboxes_list: list[list[BBox] | None] | None = None,
    ) -> list[list[PoseResult]]:
        """Run prediction on a batch of images.

        The default implementation simply loops; subclasses may
        override for batched inference.
        """
        if bboxes_list is None:
            bboxes_list = [None] * len(images)
        return [
            self.predict(img, bbs)
            for img, bbs in zip(images, bboxes_list)
        ]
