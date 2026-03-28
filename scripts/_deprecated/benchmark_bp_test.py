"""Benchmark all models on Bizarre Pose test (487 images).

Uses the same evaluation method as the Bizarre Pose paper (WACV 2022):
per-keypoint OKS correctness rate with COCO sigmas.

Includes:
- Public pretrained models (COCO-only, HumanArt)
- Our fine-tuned models
- Paper's reported scores for comparison
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

# RTMPose-m config (used for both COCO and HumanArt pretrained)
RTMPOSE_M_CFG = "pose_estimation/models/configs/experiments/stages/rtmpose_m_stage_a.py"
# RTMPose-l config
RTMPOSE_L_CFG = "pose_estimation/models/configs/models/rtmpose_l_humanart.py"

MODELS = {
    # Public pretrained (no Bizarre Pose fine-tuning)
    "RTMPose-m COCO (pretrained)": (RTMPOSE_M_CFG,
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"),
    "RTMPose-l COCO (pretrained)": (RTMPOSE_L_CFG,
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"),
    "RTMPose-m HumanArt (pretrained)": (RTMPOSE_M_CFG,
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_8xb256-420e_humanart-256x192-8430627b_20230611.pth"),
    "RTMPose-l HumanArt (pretrained)": (RTMPOSE_L_CFG,
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.pth"),

    # Our fine-tuned models
    "Stage A (COCO->BP)": (RTMPOSE_M_CFG,
        "experiments/train/rtmpose_m_stage_a/best_coco_AP_epoch_10.pth"),
    "HumanArt->BP": ("pose_estimation/models/configs/models/rtmpose_m_humanart_pretrained.py",
        "experiments/train/rtmpose_m_humanart_finetune/best_coco_AP_epoch_10.pth"),
    "Curriculum S2": ("pose_estimation/models/configs/experiments/curriculum/humanart_curriculum_s2.py",
        "experiments/train/curriculum_s2/best_coco_AP_epoch_10.pth"),
}


def compute_paper_oks(gt_kps, pred_kps, bbox, thresh):
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


def evaluate(model, gt_data, image_root):
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
        oks50_scores.append(compute_paper_oks(ann["keypoints"], pred_kps, ann["bbox"], 0.5))
        oks75_scores.append(compute_paper_oks(ann["keypoints"], pred_kps, ann["bbox"], 0.75))
    return np.mean(oks50_scores), np.mean(oks75_scores), len(oks50_scores)


def main():
    project_root = Path(__file__).resolve().parent.parent
    gt_file = project_root / "data" / "bizarre_pose" / "coco" / "test.json"
    image_root = project_root / "data" / "bizarre_pose" / "raw" / "bizarre_pose_dataset" / "raw"

    with open(gt_file) as f:
        gt_data = json.load(f)

    # Ensure bbox from keypoints
    for ann in gt_data["annotations"]:
        kps = ann["keypoints"]
        xs = [kps[i*3] for i in range(17) if int(kps[i*3+2]) > 0]
        ys = [kps[i*3+1] for i in range(17) if int(kps[i*3+2]) > 0]
        if xs:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w, h = x_max - x_min, y_max - y_min
            pad_x, pad_y = w * 0.1, h * 0.1
            ann["bbox"] = [x_min - pad_x, y_min - pad_y, w + 2*pad_x, h + 2*pad_y]

    print("=" * 75)
    print("Bizarre Pose Test Benchmark (487 images, paper-compatible evaluation)")
    print("=" * 75)
    print()
    print("Paper reference: Chen & Zwicker, WACV 2022")
    print("  Feature Concat +new data:  OKS@50=0.898  OKS@75=0.793")
    print("  Feature Matching +new data: OKS@50=0.895  OKS@75=0.791")
    print("  Task-Pretrained R-CNN:      OKS@50=0.758  OKS@75=0.672")
    print()
    print(f"{'Model':<38} {'OKS@50':>8} {'OKS@75':>8} {'n':>5}")
    print("-" * 75)

    for name, (cfg, ckpt) in MODELS.items():
        cfg_path = str(project_root / cfg)
        ckpt_path = ckpt if ckpt.startswith("http") else str(project_root / ckpt)
        try:
            model = RTMPoseEstimator(config=cfg_path, checkpoint=ckpt_path, device="cpu")
            oks50, oks75, n = evaluate(model, gt_data, image_root)
            print(f"{name:<38} {oks50:>8.3f} {oks75:>8.3f} {n:>5}")
        except Exception as e:
            print(f"{name:<38} ERROR: {e}")

    print()
    print("OKS@50/75 = fraction of keypoints with per-keypoint OKS >= threshold")
    print("(same method as Bizarre Pose paper, NOT COCO AP)")


if __name__ == "__main__":
    main()
