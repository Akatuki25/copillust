"""Abstract base class for pose evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pose_estimation.core.types import PoseResult


class BaseEvaluator(ABC):
    """Interface for evaluating pose estimation results.

    Subclasses implement concrete metrics (OKS AP, PCK, collapse rate,
    left-right confusion, etc.) and can filter by metadata subset.
    """

    @abstractmethod
    def evaluate(
        self,
        predictions: list[PoseResult],
        ground_truth: list[PoseResult],
        subset_filter: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Compute evaluation metrics.

        Args:
            predictions: Model outputs.
            ground_truth: Ground-truth annotations.
            subset_filter: Optional metadata filter
                (e.g. ``{"body_type": "chibi"}``).  Only GT/pred pairs
                whose GT metadata matches *all* filter keys are included.

        Returns:
            Dictionary of metric_name → value.
        """

    @staticmethod
    def filter_by_metadata(
        predictions: list[PoseResult],
        ground_truth: list[PoseResult],
        subset_filter: dict[str, Any],
    ) -> tuple[list[PoseResult], list[PoseResult]]:
        """Keep only (pred, gt) pairs whose GT metadata matches the filter."""
        filtered_preds: list[PoseResult] = []
        filtered_gt: list[PoseResult] = []
        for pred, gt in zip(predictions, ground_truth):
            if all(gt.metadata.get(k) == v for k, v in subset_filter.items()):
                filtered_preds.append(pred)
                filtered_gt.append(gt)
        return filtered_preds, filtered_gt
