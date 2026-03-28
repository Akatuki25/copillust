"""Visualization utilities for pose estimation results.

Draws skeletons, keypoints, and bounding boxes on images.
Supports side-by-side comparison (baseline vs fine-tuned).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pose_estimation.core.constants import (
    COCO17_KEYPOINTS,
    COCO17_SKELETON,
    KEYPOINT_COLORS,
    SKELETON_COLORS,
)
from pose_estimation.core.types import PoseResult


def draw_pose_on_image(
    image: np.ndarray,
    poses: list[PoseResult],
    confidence_threshold: float = 0.3,
    keypoint_radius: int = 4,
    line_thickness: int = 2,
    draw_bbox: bool = True,
    draw_labels: bool = False,
) -> np.ndarray:
    """Draw pose estimation results on an image.

    Args:
        image: BGR uint8 image (will be copied, not modified in-place).
        poses: List of PoseResult to draw.
        confidence_threshold: Only draw keypoints above this confidence.
        keypoint_radius: Circle radius for keypoints.
        line_thickness: Line width for skeleton edges.
        draw_bbox: Whether to draw bounding boxes.
        draw_labels: Whether to draw keypoint name labels.

    Returns:
        Annotated image (BGR uint8).
    """
    vis = image.copy()

    for pose in poses:
        kps = pose.keypoints_array()  # (17, 3): x, y, vis

        # Draw bounding box.
        if draw_bbox:
            x1, y1, x2, y2 = [int(v) for v in pose.bbox.to_xyxy()]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
            if pose.score > 0:
                cv2.putText(
                    vis, f"{pose.score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                )

        # Draw skeleton lines.
        for edge_idx, (i, j) in enumerate(COCO17_SKELETON):
            if (kps[i, 2] > 0 and kps[j, 2] > 0 and
                    pose.keypoints[i].confidence > confidence_threshold and
                    pose.keypoints[j].confidence > confidence_threshold):
                pt1 = (int(kps[i, 0]), int(kps[i, 1]))
                pt2 = (int(kps[j, 0]), int(kps[j, 1]))
                color = SKELETON_COLORS[edge_idx % len(SKELETON_COLORS)]
                # Convert RGB to BGR for OpenCV.
                color_bgr = (color[2], color[1], color[0])
                cv2.line(vis, pt1, pt2, color_bgr, line_thickness)

        # Draw keypoints.
        for i in range(len(pose.keypoints)):
            kp = pose.keypoints[i]
            if kp.visibility > 0 and kp.confidence > confidence_threshold:
                pt = (int(kp.x), int(kp.y))
                color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
                color_bgr = (color[2], color[1], color[0])

                # Confidence-based opacity via circle fill.
                alpha = min(1.0, kp.confidence)
                overlay = vis.copy()
                cv2.circle(overlay, pt, keypoint_radius, color_bgr, -1)
                cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

                cv2.circle(vis, pt, keypoint_radius, color_bgr, 1)

                if draw_labels:
                    label = COCO17_KEYPOINTS[i]
                    cv2.putText(
                        vis, label, (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_bgr, 1,
                    )

    return vis


def compare_side_by_side(
    image: np.ndarray,
    poses_a: list[PoseResult],
    poses_b: list[PoseResult],
    label_a: str = "Baseline",
    label_b: str = "Fine-tuned",
) -> np.ndarray:
    """Create a side-by-side comparison of two pose predictions.

    Args:
        image: Original BGR image.
        poses_a: First set of predictions (e.g., baseline).
        poses_b: Second set of predictions (e.g., fine-tuned).
        label_a: Label for the first prediction.
        label_b: Label for the second prediction.

    Returns:
        Horizontally concatenated comparison image.
    """
    vis_a = draw_pose_on_image(image, poses_a)
    vis_b = draw_pose_on_image(image, poses_b)

    # Add labels.
    h = vis_a.shape[0]
    cv2.putText(vis_a, label_a, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(vis_b, label_b, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Add a separator line.
    separator = np.ones((h, 3, 3), dtype=np.uint8) * 128
    return np.hstack([vis_a, separator, vis_b])


def save_comparison_grid(
    image_paths: list[str | Path],
    results_a: dict[str, list[PoseResult]],
    results_b: dict[str, list[PoseResult]],
    output_dir: str | Path,
    label_a: str = "Baseline",
    label_b: str = "Fine-tuned",
) -> None:
    """Save comparison images for multiple files.

    Args:
        image_paths: List of image file paths.
        results_a: Filename → poses mapping for first model.
        results_b: Filename → poses mapping for second model.
        output_dir: Directory to save comparison images.
        label_a: Label for first model.
        label_b: Label for second model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        img_path = Path(img_path)
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Skipping {img_path}: could not read")
            continue

        fname = img_path.name
        poses_a = results_a.get(fname, [])
        poses_b = results_b.get(fname, [])

        comp = compare_side_by_side(image, poses_a, poses_b, label_a, label_b)
        out_path = output_dir / f"compare_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), comp)

    print(f"Comparison images saved to {output_dir}")
