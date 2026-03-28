"""Evaluate models against mydata GT using standard COCO AP (OKS-based).

Uses pycocotools COCOeval with iouType='keypoints'.
This is the same metric used by Bizarre Pose (WACV 2022) and HumanArt (CVPR 2023).

Usage:
    python scripts/evaluate_with_gt.py
"""

from __future__ import annotations

import json
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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pose_estimation.models.rtmpose_estimator import RTMPoseEstimator

CONFIGS = {
    "StageA": ("pose_estimation/models/configs/experiments/stages/rtmpose_m_stage_a.py",
               "experiments/train/rtmpose_m_stage_a/best_coco_AP_epoch_10.pth"),
    "HumanArt": ("pose_estimation/models/configs/models/rtmpose_m_humanart_pretrained.py",
                 "experiments/train/rtmpose_m_humanart_finetune/best_coco_AP_epoch_10.pth"),
    "HRNet48": ("pose_estimation/models/configs/models/hrnet_w48_bizarre_pose.py",
                "experiments/train/hrnet_w48_bizarre_pose/best_coco_AP_epoch_10.pth"),
    "HRNet-DK": ("pose_estimation/models/configs/models/hrnet_w48_dark_bizarre_pose.py",
                 "experiments/train/hrnet_w48_dark/best_coco_AP_epoch_10.pth"),
    "curricul": ("pose_estimation/models/configs/experiments/curriculum/humanart_curriculum_s2.py",
                 "experiments/train/curriculum_s2/best_coco_AP_epoch_10.pth"),
    "3L+mask": ("pose_estimation/models/configs/experiments/techniques/occluded_3layer.py",
                "experiments/train/techniques/occluded_3layer/best_coco_AP_epoch_10.pth"),
    "3L_noM": ("pose_estimation/models/configs/experiments/techniques/gau_3layer_no_mask.py",
               "experiments/train/techniques/gau_3layer_no_mask/best_coco_AP_epoch_10.pth"),
    "3L_curr": ("pose_estimation/models/configs/experiments/techniques/occ3l_curriculum_s2.py",
                "experiments/train/techniques/occ3l_curriculum_s2/best_coco_AP_epoch_8.pth"),
}


def run_inference(model, gt_data, mydata_dir):
    """Run inference and return results in COCO format."""
    results = []
    for img_info in gt_data["images"]:
        img_path = mydata_dir / img_info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        preds = model.predict(image)
        if not preds:
            continue

        keypoints = []
        score_sum = 0
        for kp in preds[0].keypoints:
            keypoints.extend([float(kp.x), float(kp.y), 2])
            score_sum += kp.confidence

        results.append({
            "image_id": img_info["id"],
            "category_id": 1,
            "keypoints": keypoints,
            "score": float(score_sum / len(preds[0].keypoints)),
        })

    return results


def evaluate_coco_ap(gt_file, results):
    """Compute COCO AP using pycocotools."""
    coco_gt = COCO(str(gt_file))

    if not results:
        print("  No results to evaluate")
        return {}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "AP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "AR": coco_eval.stats[5],
    }


def ensure_coco_format(gt_file):
    """Ensure GT file has proper COCO format with categories."""
    with open(gt_file) as f:
        data = json.load(f)

    if "categories" not in data or not data["categories"]:
        data["categories"] = [{
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ],
            "skeleton": [
                [16,14],[14,12],[17,15],[15,13],[12,13],
                [6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],
                [2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
            ]
        }]
        with open(gt_file, "w") as f:
            json.dump(data, f)

    # Ensure each annotation has required fields
    modified = False
    for ann in data["annotations"]:
        if "area" not in ann or ann["area"] == 0:
            # Compute area from bbox or image size
            img = next((i for i in data["images"] if i["id"] == ann["image_id"]), None)
            if img:
                ann["area"] = img["width"] * img["height"]
                modified = True
        if "iscrowd" not in ann:
            ann["iscrowd"] = 0
            modified = True

    if modified:
        with open(gt_file, "w") as f:
            json.dump(data, f)


def main():
    project_root = Path(__file__).resolve().parent.parent
    mydata_dir = project_root / "mydata"
    gt_file = mydata_dir / "annotations.json"

    ensure_coco_format(gt_file)

    print("=" * 70)
    print("COCO AP Evaluation (OKS-based, standard metric)")
    print("=" * 70)

    all_results = {}

    for name, (cfg, ckpt) in CONFIGS.items():
        cfg_path = str(project_root / cfg)
        ckpt_path = str(project_root / ckpt)

        print(f"\n--- {name} ---")
        print(f"Loading model...")
        model = RTMPoseEstimator(config=cfg_path, checkpoint=ckpt_path, device="cpu")

        results = run_inference(model, json.loads(gt_file.read_text()), mydata_dir)
        metrics = evaluate_coco_ap(gt_file, results)
        all_results[name] = metrics

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Model':>12} {'AP':>8} {'AP50':>8} {'AP75':>8} {'AR':>8}")
    print("-" * 70)
    for name, m in all_results.items():
        if m:
            print(f"{name:>12} {m['AP']:>8.3f} {m['AP50']:>8.3f} {m['AP75']:>8.3f} {m['AR']:>8.3f}")
        else:
            print(f"{name:>12}     N/A")


if __name__ == "__main__":
    main()
