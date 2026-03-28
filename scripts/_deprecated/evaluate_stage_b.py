"""Evaluate Stage B model on mydata and generate comparison images (Stage A vs Stage B)."""

from __future__ import annotations

import sys
from pathlib import Path

# Patch torch.load for PyTorch 2.6+
import torch
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pose_estimation.models.rtmpose_estimator import RTMPoseEstimator
from pose_estimation.core.constants import COCO17_SKELETON

# COCO17 skeleton connections (0-indexed pairs)
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
    (200, 100, 50), (50, 100, 200), (100, 200, 50), (200, 50, 100),
    (50, 200, 100),
]


def draw_skeleton(image: np.ndarray, keypoints, conf_threshold=0.3) -> tuple[np.ndarray, int, int]:
    """Draw skeleton on image. Returns (image, high_conf_count, total)."""
    vis = image.copy()
    h, w = vis.shape[:2]
    high_conf = 0

    for kp in keypoints:
        if kp.confidence > conf_threshold:
            high_conf += 1
            cx, cy = int(kp.x), int(kp.y)
            cv2.circle(vis, (cx, cy), max(3, w // 150), (0, 255, 0), -1)
        elif kp.confidence > 0.1:
            cx, cy = int(kp.x), int(kp.y)
            cv2.circle(vis, (cx, cy), max(2, w // 200), (0, 128, 255), -1)

    for i, j in SKELETON_PAIRS:
        if i < len(keypoints) and j < len(keypoints):
            kp1, kp2 = keypoints[i], keypoints[j]
            if kp1.confidence > conf_threshold and kp2.confidence > conf_threshold:
                pt1 = (int(kp1.x), int(kp1.y))
                pt2 = (int(kp2.x), int(kp2.y))
                cv2.line(vis, pt1, pt2, COLORS[i % len(COLORS)], max(2, w // 300))

    return vis, high_conf, 17


def main():
    project_root = Path(__file__).resolve().parent.parent
    mydata_dir = project_root / "mydata"
    output_dir = project_root / "experiments" / "stage_b_eval"
    comparisons_dir = output_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    # Stage A config/checkpoint
    stage_a_config = str(project_root / "pose_estimation" / "models" / "configs" / "rtmpose_m_stage_a.py")
    stage_a_ckpt = str(project_root / "experiments" / "rtmpose_m_stage_a" / "best_coco_AP_epoch_10.pth")

    # Stage B config/checkpoint
    stage_b_config = str(project_root / "pose_estimation" / "models" / "configs" / "rtmpose_m_stage_b.py")
    stage_b_ckpt = str(project_root / "experiments" / "rtmpose_m_stage_b" / "best_coco_AP_epoch_50.pth")

    print("Loading Stage A model...")
    model_a = RTMPoseEstimator(config=stage_a_config, checkpoint=stage_a_ckpt, device="cpu")

    print("Loading Stage B model...")
    model_b = RTMPoseEstimator(config=stage_b_config, checkpoint=stage_b_ckpt, device="cpu")

    categories = ["chibi", "lineart", "part"]
    results = []

    for cat in categories:
        cat_dir = mydata_dir / cat
        if not cat_dir.exists():
            continue
        for img_path in sorted(cat_dir.glob("*.jpeg")) + sorted(cat_dir.glob("*.png")) + sorted(cat_dir.glob("*.jpg")):
            print(f"\nProcessing: {cat}/{img_path.name}")
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  Failed to load")
                continue

            # Stage A prediction
            preds_a = model_a.predict(image)
            if preds_a:
                vis_a, high_a, total = draw_skeleton(image, preds_a[0].keypoints)
            else:
                vis_a = image.copy()
                high_a, total = 0, 17

            # Stage B prediction
            preds_b = model_b.predict(image)
            if preds_b:
                vis_b, high_b, total = draw_skeleton(image, preds_b[0].keypoints)
            else:
                vis_b = image.copy()
                high_b, total = 0, 17

            results.append({
                "name": img_path.stem,
                "category": cat,
                "stage_a": high_a,
                "stage_b": high_b,
                "delta": high_b - high_a,
            })

            # Create comparison image: Stage A | Stage B
            h, w = image.shape[:2]
            target_h = 600
            scale = target_h / h
            target_w = int(w * scale)

            vis_a_resized = cv2.resize(vis_a, (target_w, target_h))
            vis_b_resized = cv2.resize(vis_b, (target_w, target_h))

            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(vis_a_resized, f"Stage A: {high_a}/17", (10, 30), font, 0.8, (0, 255, 0), 2)
            cv2.putText(vis_b_resized, f"Stage B: {high_b}/17", (10, 30), font, 0.8, (0, 255, 0), 2)

            comparison = np.hstack([vis_a_resized, vis_b_resized])
            out_path = comparisons_dir / f"{cat}_{img_path.stem}_ab_compare.jpg"
            cv2.imwrite(str(out_path), comparison)
            print(f"  Stage A: {high_a}/17  Stage B: {high_b}/17  delta: {high_b - high_a:+d}")

    # Print summary table
    print("\n" + "=" * 70)
    print("Stage A vs Stage B Summary")
    print("=" * 70)
    print(f"{'Image':<20} {'Category':<10} {'Stage A':<10} {'Stage B':<10} {'Delta':<10}")
    print("-" * 70)
    for r in results:
        short_name = r['name'][:18]
        print(f"{short_name:<20} {r['category']:<10} {r['stage_a']}/17      {r['stage_b']}/17      {r['delta']:+d}")

    total_a = sum(r['stage_a'] for r in results)
    total_b = sum(r['stage_b'] for r in results)
    n = len(results)
    print("-" * 70)
    print(f"{'TOTAL':<20} {'':10} {total_a}/{n*17}     {total_b}/{n*17}     {total_b - total_a:+d}")

    # Per-category summary
    for cat in ["chibi", "lineart", "part"]:
        cat_results = [r for r in results if r['category'] == cat]
        if cat_results:
            cat_a = sum(r['stage_a'] for r in cat_results)
            cat_b = sum(r['stage_b'] for r in cat_results)
            cn = len(cat_results)
            print(f"  {cat:<18} {'':10} {cat_a}/{cn*17}     {cat_b}/{cn*17}     {cat_b - cat_a:+d}")

    print(f"\nComparison images saved to: {comparisons_dir}")


if __name__ == "__main__":
    main()
