"""Shared evaluation metrics.

Contains the Bizarre Pose paper-compatible OKS evaluation
(per-keypoint correctness rate, NOT COCO AP).

This is the standard metric for comparing with:
- Chen & Zwicker, WACV 2022 (Bizarre Pose)
- Khungurn & Chou, MANPU 2016 (AnimeDrawingsDataset)
"""

import math
import numpy as np

# Standard COCO per-keypoint sigmas
COCO_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035,
    0.079, 0.079, 0.072, 0.072, 0.062, 0.062,
    0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
])

COCO17_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]


def compute_bbox_from_keypoints(keypoints, padding=0.1):
    """Compute tight bounding box from COCO-format keypoints list.

    Args:
        keypoints: flat list [x1,y1,v1, x2,y2,v2, ...]
        padding: fraction of bbox size to add as padding

    Returns:
        [x, y, w, h] or None if no visible keypoints
    """
    xs, ys = [], []
    for i in range(17):
        x, y, v = keypoints[i*3], keypoints[i*3+1], int(keypoints[i*3+2])
        if v > 0:
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w, h = x_max - x_min, y_max - y_min
    pad_x, pad_y = w * padding, h * padding
    return [x_min - pad_x, y_min - pad_y, w + 2*pad_x, h + 2*pad_y]


def paper_oks_per_image(gt_kps, pred_kps, bbox, thresh=0.5):
    """Compute OKS the same way as the Bizarre Pose paper (WACV 2022).

    Per-keypoint OKS correctness rate: fraction of visible keypoints
    whose individual OKS score >= threshold.

    NOTE: This is NOT the COCO AP metric. It is per-keypoint accuracy.

    Args:
        gt_kps: flat list [x1,y1,v1, ...] (51 values)
        pred_kps: list of (x, y) tuples for 17 keypoints
        bbox: [x, y, w, h]
        thresh: OKS threshold (0.5 or 0.75)

    Returns:
        fraction of correct keypoints (0.0 to 1.0)
    """
    bx, by, bw, bh = bbox
    max_dim = max(bw, bh)
    s = math.sqrt(bw * bh) / max_dim

    correct = 0
    total = 0

    for i in range(17):
        gx, gy, gv = gt_kps[i*3], gt_kps[i*3+1], int(gt_kps[i*3+2])
        if gv == 0:
            continue

        px, py = pred_kps[i]
        dx = (px - gx) / max_dim
        dy = (py - gy) / max_dim
        d = math.sqrt(dx**2 + dy**2)
        oks_i = math.exp(-d**2 / (2 * s**2 * COCO_SIGMAS[i]**2))

        total += 1
        if oks_i >= thresh:
            correct += 1

    return correct / total if total > 0 else 0


def paper_oks_per_keypoint(gt_kps, pred_kps, bbox, thresh=0.5):
    """Compute per-keypoint OKS correctness for a single image.

    Returns:
        dict of {keypoint_name: (correct, visible)} for visible keypoints
    """
    bx, by, bw, bh = bbox
    max_dim = max(bw, bh)
    s = math.sqrt(bw * bh) / max_dim

    results = {}
    for i in range(17):
        gx, gy, gv = gt_kps[i*3], gt_kps[i*3+1], int(gt_kps[i*3+2])
        if gv == 0:
            continue

        px, py = pred_kps[i]
        dx = (px - gx) / max_dim
        dy = (py - gy) / max_dim
        d = math.sqrt(dx**2 + dy**2)
        oks_i = math.exp(-d**2 / (2 * s**2 * COCO_SIGMAS[i]**2))

        results[COCO17_NAMES[i]] = (oks_i >= thresh, True)

    return results
