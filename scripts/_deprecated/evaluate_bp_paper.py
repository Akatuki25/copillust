"""Evaluate using the same method as the Bizarre Pose paper (WACV 2022).

Their "OKS@50" = fraction of keypoints with per-keypoint OKS >= 0.5
Their "OKS@75" = fraction of keypoints with per-keypoint OKS >= 0.75

OKS per keypoint: exp(-d^2 / (2 * s^2 * sigma^2))
where d = distance (normalized by max bbox dimension)
      s = sqrt(bbox_w * bbox_h) / max(bbox_w, bbox_h)
      sigma = COCO per-keypoint sigma

Evaluated on GT bbox, single instance per image.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pose_estimation.models.rtmpose_estimator import RTMPoseEstimator

COCO_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035,
    0.079, 0.079, 0.072, 0.072, 0.062, 0.062,
    0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
])

CONFIGS = {
    "StageA": ("pose_estimation/models/configs/experiments/stages/rtmpose_m_stage_a.py",
               "experiments/train/rtmpose_m_stage_a/best_coco_AP_epoch_10.pth"),
    "HumanArt": ("pose_estimation/models/configs/models/rtmpose_m_humanart_pretrained.py",
                 "experiments/train/rtmpose_m_humanart_finetune/best_coco_AP_epoch_10.pth"),
    "curricul": ("pose_estimation/models/configs/experiments/curriculum/humanart_curriculum_s2.py",
                 "experiments/train/curriculum_s2/best_coco_AP_epoch_10.pth"),
    "3L+mask": ("pose_estimation/models/configs/experiments/techniques/occluded_3layer.py",
                "experiments/train/techniques/occluded_3layer/best_coco_AP_epoch_10.pth"),
    "3L_curr": ("pose_estimation/models/configs/experiments/techniques/occ3l_curriculum_s2.py",
                "experiments/train/techniques/occ3l_curriculum_s2/best_coco_AP_epoch_8.pth"),
}


def compute_paper_oks(gt_kps, pred_kps, bbox, thresh=0.5):
    """Compute OKS the same way as the Bizarre Pose paper.

    Args:
        gt_kps: list of 51 values (x,y,v)*17
        pred_kps: list of (x,y) tuples for 17 keypoints
        bbox: [x, y, w, h]
        thresh: OKS threshold

    Returns:
        correct_fraction: fraction of visible keypoints with OKS >= thresh
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

        # Normalize distance by max bbox dimension (paper convention)
        dx = (px - gx) / max_dim
        dy = (py - gy) / max_dim
        d = math.sqrt(dx**2 + dy**2)

        oks_i = math.exp(-d**2 / (2 * s**2 * COCO_SIGMAS[i]**2))

        total += 1
        if oks_i >= thresh:
            correct += 1

    return correct / total if total > 0 else 0


def evaluate_split(model, gt_data, image_root):
    """Evaluate on a dataset split."""
    oks50_scores = []
    oks75_scores = []

    for img_info in gt_data["images"]:
        ann = next((a for a in gt_data["annotations"] if a["image_id"] == img_info["id"]), None)
        if ann is None:
            continue

        img_path = image_root / img_info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        preds = model.predict(image)
        if not preds:
            continue

        pred_kps = [(kp.x, kp.y) for kp in preds[0].keypoints]
        gt_kps = ann["keypoints"]
        bbox = ann["bbox"]

        oks50 = compute_paper_oks(gt_kps, pred_kps, bbox, thresh=0.5)
        oks75 = compute_paper_oks(gt_kps, pred_kps, bbox, thresh=0.75)

        oks50_scores.append(oks50)
        oks75_scores.append(oks75)

    return {
        "OKS@50": np.mean(oks50_scores) if oks50_scores else 0,
        "OKS@75": np.mean(oks75_scores) if oks75_scores else 0,
        "n": len(oks50_scores),
    }


def main():
    project_root = Path(__file__).resolve().parent.parent

    # Bizarre Pose test split
    bp_gt_file = project_root / "data" / "bizarre_pose" / "coco" / "test.json"
    bp_image_root = project_root / "data" / "bizarre_pose" / "raw" / "bizarre_pose_dataset" / "raw"

    with open(bp_gt_file) as f:
        bp_gt = json.load(f)

    # Ensure bbox exists (from keypoints)
    for ann in bp_gt["annotations"]:
        kps = ann["keypoints"]
        xs = [kps[i*3] for i in range(17) if int(kps[i*3+2]) > 0]
        ys = [kps[i*3+1] for i in range(17) if int(kps[i*3+2]) > 0]
        if xs:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w, h = x_max - x_min, y_max - y_min
            pad_x, pad_y = w * 0.1, h * 0.1
            ann["bbox"] = [x_min - pad_x, y_min - pad_y, w + 2*pad_x, h + 2*pad_y]

    # mydata
    mydata_gt_file = project_root / "mydata" / "annotations.json"
    mydata_dir = project_root / "mydata"

    with open(mydata_gt_file) as f:
        mydata_gt = json.load(f)

    print("=" * 80)
    print("Bizarre Pose Paper Evaluation Method (per-keypoint OKS correctness rate)")
    print("=" * 80)
    print()
    print("Paper best (Feature Concat +new data): OKS@50=0.898  OKS@75=0.793")
    print()

    header = f"{'Model':>12} | {'BP test OKS@50':>14} {'OKS@75':>8} | {'mydata OKS@50':>14} {'OKS@75':>8}"
    print(header)
    print("-" * 80)

    for name, (cfg, ckpt) in CONFIGS.items():
        cfg_path = str(project_root / cfg)
        ckpt_path = str(project_root / ckpt)
        model = RTMPoseEstimator(config=cfg_path, checkpoint=ckpt_path, device="cpu")

        bp_result = evaluate_split(model, bp_gt, bp_image_root)
        my_result = evaluate_split(model, mydata_gt, mydata_dir)

        print(f"{name:>12} | {bp_result['OKS@50']:>13.3f} {bp_result['OKS@75']:>8.3f} | {my_result['OKS@50']:>13.3f} {my_result['OKS@75']:>8.3f}")


if __name__ == "__main__":
    main()
