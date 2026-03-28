"""Evaluate models on Bizarre Pose test split (487 images) using COCO AP.

This enables direct comparison with the Bizarre Pose paper (WACV 2022).
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
    "curricul": ("pose_estimation/models/configs/experiments/curriculum/humanart_curriculum_s2.py",
                 "experiments/train/curriculum_s2/best_coco_AP_epoch_10.pth"),
    "3L+mask": ("pose_estimation/models/configs/experiments/techniques/occluded_3layer.py",
                "experiments/train/techniques/occluded_3layer/best_coco_AP_epoch_10.pth"),
    "3L_curr": ("pose_estimation/models/configs/experiments/techniques/occ3l_curriculum_s2.py",
                "experiments/train/techniques/occ3l_curriculum_s2/best_coco_AP_epoch_8.pth"),
}


def run_inference(model, gt_data, image_root):
    """Run inference and return COCO-format results."""
    results = []
    for img_info in gt_data["images"]:
        img_path = image_root / img_info["file_name"]
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


def main():
    project_root = Path(__file__).resolve().parent.parent
    gt_file = project_root / "data" / "bizarre_pose" / "coco" / "test.json"
    image_root = project_root / "data" / "bizarre_pose" / "raw" / "bizarre_pose_dataset" / "raw"

    # Ensure annotations have proper bbox/area
    with open(gt_file) as f:
        gt_data = json.load(f)

    # Add categories if missing
    if "categories" not in gt_data or not gt_data["categories"]:
        gt_data["categories"] = [{
            "supercategory": "person", "id": 1, "name": "person",
            "keypoints": [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ],
            "skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],
                         [6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],
                         [2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
        }]

    # Compute tight bbox from keypoints for each annotation
    for ann in gt_data["annotations"]:
        kps = ann["keypoints"]
        xs, ys = [], []
        for i in range(17):
            x, y, v = kps[i*3], kps[i*3+1], int(kps[i*3+2])
            if v > 0:
                xs.append(x)
                ys.append(y)
        if xs:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w, h = x_max - x_min, y_max - y_min
            pad_x, pad_y = w * 0.1, h * 0.1
            ann["bbox"] = [round(x_min - pad_x, 1), round(y_min - pad_y, 1),
                           round(w + 2*pad_x, 1), round(h + 2*pad_y, 1)]
            ann["area"] = round(ann["bbox"][2] * ann["bbox"][3], 1)
        if "iscrowd" not in ann:
            ann["iscrowd"] = 0

    # Save fixed version
    fixed_gt = project_root / "data" / "bizarre_pose" / "coco" / "test_eval.json"
    with open(fixed_gt, "w") as f:
        json.dump(gt_data, f)

    print("=" * 70)
    print(f"Bizarre Pose Test Set Evaluation ({len(gt_data['images'])} images)")
    print("COCO AP (OKS-based) - comparable to Bizarre Pose paper (WACV 2022)")
    print("=" * 70)

    all_results = {}

    for name, (cfg, ckpt) in CONFIGS.items():
        cfg_path = str(project_root / cfg)
        ckpt_path = str(project_root / ckpt)

        print(f"\n--- {name} ---")
        model = RTMPoseEstimator(config=cfg_path, checkpoint=ckpt_path, device="cpu")
        results = run_inference(model, gt_data, image_root)

        coco_gt = COCO(str(fixed_gt))
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        all_results[name] = {
            "AP": coco_eval.stats[0],
            "AP50": coco_eval.stats[1],
            "AP75": coco_eval.stats[2],
            "AR": coco_eval.stats[5],
        }

    print("\n" + "=" * 70)
    print(f"{'Model':>12} {'AP':>8} {'AP50':>8} {'AP75':>8} {'AR':>8}")
    print("-" * 70)
    for name, m in all_results.items():
        print(f"{name:>12} {m['AP']:>8.3f} {m['AP50']:>8.3f} {m['AP75']:>8.3f} {m['AR']:>8.3f}")


if __name__ == "__main__":
    main()
