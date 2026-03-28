"""Pose estimation metrics: PCK, OKS AP, collapse rate, left-right confusion.

All metrics operate on lists of PoseResult (predictions and ground truth).
Metrics are computed per-keypoint and aggregated, matching the evaluation
requirements from docs/task.md.
"""

from __future__ import annotations

import numpy as np

from pose_estimation.core.constants import (
    COCO17_SIGMAS,
    COCO17_SWAP_PAIRS,
    NUM_KEYPOINTS,
)
from pose_estimation.core.types import PoseResult


def pck(
    predictions: list[PoseResult],
    ground_truth: list[PoseResult],
    threshold: float = 0.2,
    normalize_by: str = "bbox",
) -> dict[str, float]:
    """Compute Percentage of Correct Keypoints (PCK).

    Only evaluates keypoints that are labeled (visibility > 0) in GT.

    Args:
        predictions: Model predictions.
        ground_truth: Ground truth annotations.
        threshold: Distance threshold as fraction of normalization factor.
        normalize_by: "bbox" (bbox diagonal) or "head" (head size).

    Returns:
        Dict with "pck_overall" and "pck_visible_only".
    """
    correct = 0
    total = 0
    correct_visible = 0
    total_visible = 0

    for pred, gt in zip(predictions, ground_truth):
        pred_arr = pred.keypoints_array()
        gt_arr = gt.keypoints_array()

        # Normalization factor.
        if normalize_by == "bbox":
            norm = np.sqrt(gt.bbox.w ** 2 + gt.bbox.h ** 2)
        else:
            norm = max(gt.bbox.h, gt.bbox.w)
        if norm < 1e-6:
            continue

        for i in range(NUM_KEYPOINTS):
            gt_vis = int(gt_arr[i, 2])
            if gt_vis == 0:  # Unlabeled — skip.
                continue

            dist = np.sqrt((pred_arr[i, 0] - gt_arr[i, 0]) ** 2 +
                           (pred_arr[i, 1] - gt_arr[i, 1]) ** 2)
            is_correct = dist / norm < threshold

            total += 1
            if is_correct:
                correct += 1

            if gt_vis == 2:  # Visible only.
                total_visible += 1
                if is_correct:
                    correct_visible += 1

    return {
        "pck_all_labeled": correct / total if total > 0 else 0.0,
        "pck_visible_only": correct_visible / total_visible if total_visible > 0 else 0.0,
        "pck_total_kpts": total,
        "pck_total_visible_kpts": total_visible,
    }


def oks_ap(
    predictions: list[PoseResult],
    ground_truth: list[PoseResult],
    sigmas: list[float] | None = None,
) -> dict[str, float]:
    """Compute OKS-based Average Precision (simplified).

    This is a simplified version that computes mean OKS across all pairs.
    For full COCO AP evaluation, use pycocotools.

    Args:
        predictions: Model predictions.
        ground_truth: Ground truth annotations.
        sigmas: Per-keypoint sigma values (default: COCO17 sigmas).

    Returns:
        Dict with mean OKS and approximate AP at various thresholds.
    """
    if sigmas is None:
        sigmas = COCO17_SIGMAS

    sigmas_arr = np.array(sigmas, dtype=np.float64)
    oks_scores: list[float] = []

    for pred, gt in zip(predictions, ground_truth):
        pred_arr = pred.keypoints_array()
        gt_arr = gt.keypoints_array()
        area = gt.bbox.area
        if area < 1e-6:
            continue

        # OKS computation.
        visible_mask = gt_arr[:, 2] > 0
        if not visible_mask.any():
            continue

        dx = pred_arr[:, 0] - gt_arr[:, 0]
        dy = pred_arr[:, 1] - gt_arr[:, 1]
        d_sq = dx ** 2 + dy ** 2
        s_sq = (2 * sigmas_arr) ** 2
        exp_term = np.exp(-d_sq / (2 * area * s_sq + 1e-8))

        oks = float(np.sum(exp_term[visible_mask]) / np.sum(visible_mask))
        oks_scores.append(oks)

    if not oks_scores:
        return {"mean_oks": 0.0, "ap_50": 0.0, "ap_75": 0.0}

    oks_arr = np.array(oks_scores)
    return {
        "mean_oks": float(oks_arr.mean()),
        "ap_50": float((oks_arr >= 0.5).mean()),
        "ap_75": float((oks_arr >= 0.75).mean()),
    }


def collapse_rate(predictions: list[PoseResult], min_visible: int = 3) -> dict[str, float]:
    """Compute the rate of collapsed/failed predictions.

    A prediction is considered "collapsed" if fewer than min_visible
    keypoints have confidence above a threshold.

    Args:
        predictions: Model predictions.
        min_visible: Minimum keypoints required for a non-collapsed prediction.

    Returns:
        Dict with collapse rate and count.
    """
    collapsed = 0
    total = len(predictions)

    for pred in predictions:
        high_conf = sum(1 for kp in pred.keypoints if kp.confidence > 0.3)
        if high_conf < min_visible:
            collapsed += 1

    return {
        "collapse_rate": collapsed / total if total > 0 else 0.0,
        "collapsed_count": collapsed,
        "total_predictions": total,
    }


def left_right_confusion_rate(
    predictions: list[PoseResult],
    ground_truth: list[PoseResult],
    distance_threshold: float = 0.1,
) -> dict[str, float]:
    """Detect left-right swaps in predicted keypoints.

    For each swap pair (e.g., left_shoulder/right_shoulder), checks if
    the prediction is closer to the opposite GT keypoint than the correct one.

    Args:
        predictions: Model predictions.
        ground_truth: Ground truth annotations.
        distance_threshold: Min distance (as fraction of bbox diagonal) to count.

    Returns:
        Dict with confusion rate and count.
    """
    confused = 0
    total_pairs = 0

    for pred, gt in zip(predictions, ground_truth):
        pred_arr = pred.keypoints_array()
        gt_arr = gt.keypoints_array()
        norm = np.sqrt(gt.bbox.w ** 2 + gt.bbox.h ** 2)
        if norm < 1e-6:
            continue

        for left_idx, right_idx in COCO17_SWAP_PAIRS:
            # Both must be labeled in GT.
            if gt_arr[left_idx, 2] == 0 or gt_arr[right_idx, 2] == 0:
                continue

            total_pairs += 1

            # Distance from predicted left to GT left vs GT right.
            d_correct_left = np.sqrt(
                (pred_arr[left_idx, 0] - gt_arr[left_idx, 0]) ** 2 +
                (pred_arr[left_idx, 1] - gt_arr[left_idx, 1]) ** 2
            )
            d_swapped_left = np.sqrt(
                (pred_arr[left_idx, 0] - gt_arr[right_idx, 0]) ** 2 +
                (pred_arr[left_idx, 1] - gt_arr[right_idx, 1]) ** 2
            )

            # It's confused if predicted left is closer to GT right.
            if d_swapped_left < d_correct_left and d_correct_left / norm > distance_threshold:
                confused += 1

    return {
        "lr_confusion_rate": confused / total_pairs if total_pairs > 0 else 0.0,
        "lr_confused_pairs": confused,
        "lr_total_pairs": total_pairs,
    }


def out_of_frame_misclassification(
    predictions: list[PoseResult],
    ground_truth: list[PoseResult],
    confidence_threshold: float = 0.3,
) -> dict[str, float]:
    """Detect cases where out-of-frame GT keypoints are predicted as visible.

    This measures the model's ability to recognize that a keypoint
    should not be predicted (e.g., legs in a upper-body crop).

    Note: Requires GT visibility to distinguish between truly out-of-frame
    (visibility=0 for unlabeled, or custom kp_states).  For standard COCO
    visibility, we treat visibility=0 (unlabeled) as potentially out-of-frame.

    Args:
        predictions: Model predictions.
        ground_truth: Ground truth with out-of-frame keypoints marked.
        confidence_threshold: Prediction confidence above which we consider it "predicted visible".

    Returns:
        Dict with misclassification rate.
    """
    false_visible = 0
    total_oof = 0

    for pred, gt in zip(predictions, ground_truth):
        for i in range(NUM_KEYPOINTS):
            gt_vis = gt.keypoints[i].visibility
            if gt_vis == 0:  # Unlabeled / out-of-frame.
                total_oof += 1
                pred_conf = pred.keypoints[i].confidence
                if pred_conf > confidence_threshold:
                    false_visible += 1

    return {
        "oof_misclass_rate": false_visible / total_oof if total_oof > 0 else 0.0,
        "oof_false_visible": false_visible,
        "oof_total": total_oof,
    }
