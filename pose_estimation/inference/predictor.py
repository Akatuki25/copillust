"""High-level prediction interface using any BaseEstimator.

Provides convenience methods for single image, batch, and directory inference
that work with any estimator implementation.

Usage:
    from pose_estimation.inference.predictor import Predictor
    from pose_estimation.models.rtmpose_estimator import RTMPoseEstimator

    estimator = RTMPoseEstimator(config=..., checkpoint=...)
    predictor = Predictor(estimator)

    # Single image
    results = predictor.predict_file("path/to/image.jpg")

    # Directory
    all_results = predictor.predict_directory("path/to/images/")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pose_estimation.core.base_estimator import BaseEstimator
from pose_estimation.core.types import BBox, PoseResult


class Predictor:
    """Model-agnostic prediction wrapper.

    Takes any BaseEstimator and provides file/directory-level inference
    with consistent output format.
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def predict_image(
        self,
        image: np.ndarray,
        bboxes: list[BBox] | None = None,
    ) -> list[PoseResult]:
        """Run prediction on a BGR numpy image."""
        return self.estimator.predict(image, bboxes)

    def predict_file(
        self,
        path: str | Path,
        bboxes: list[BBox] | None = None,
    ) -> list[PoseResult]:
        """Load an image file and run prediction."""
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return self.predict_image(image, bboxes)

    def predict_directory(
        self,
        directory: str | Path,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
        bboxes_map: dict[str, list[BBox]] | None = None,
    ) -> dict[str, list[PoseResult]]:
        """Run prediction on all images in a directory.

        Args:
            directory: Path to image directory.
            extensions: File extensions to include.
            bboxes_map: Optional mapping of filename → bboxes.

        Returns:
            Dict mapping filename → list of PoseResult.
        """
        directory = Path(directory)
        results: dict[str, list[PoseResult]] = {}

        image_files = sorted(
            f for f in directory.rglob("*")
            if f.suffix.lower() in extensions
        )

        for img_path in image_files:
            rel_name = str(img_path.relative_to(directory))
            bboxes = (bboxes_map or {}).get(rel_name)
            try:
                results[rel_name] = self.predict_file(img_path, bboxes)
            except Exception as e:
                print(f"  Warning: failed on {rel_name}: {e}")
                results[rel_name] = []

        return results
