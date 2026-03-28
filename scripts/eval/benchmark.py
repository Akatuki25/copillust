"""Unified benchmark script.

Evaluates models on Bizarre Pose test set using the paper-compatible
OKS metric (per-keypoint correctness rate).

Usage:
    # Evaluate all models
    python -m scripts.eval.benchmark

    # Evaluate specific models
    python -m scripts.eval.benchmark --models "Stage A" "Curriculum S2"

    # List available models
    python -m scripts.eval.benchmark --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
_orig = torch.load
def _patched(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig(*a, **kw)
torch.load = _patched

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.eval.model_registry import ALL_MODELS, get_model, list_models
from scripts.eval.metrics import compute_bbox_from_keypoints, paper_oks_per_image
from pose_estimation.models.rtmpose_estimator import RTMPoseEstimator

# Paper reference scores
PAPER_SCORES = {
    "Feature Concat +new": {"OKS@50": 0.898, "OKS@75": 0.793},
    "Feature Matching +new": {"OKS@50": 0.895, "OKS@75": 0.791},
    "Task-Pretrained R-CNN": {"OKS@50": 0.758, "OKS@75": 0.672},
}

BP_TEST = Path("data/bizarre_pose/coco/test.json")
BP_IMAGE_ROOT = Path("data/bizarre_pose/raw/bizarre_pose_dataset/raw")
MYDATA_GT = Path("mydata/annotations.json")
MYDATA_ROOT = Path("mydata")


def evaluate_on_dataset(model, gt_data, image_root):
    """Evaluate a model on a dataset, return OKS@50 and OKS@75."""
    oks50, oks75 = [], []
    for img_info in gt_data["images"]:
        ann = next((a for a in gt_data["annotations"]
                    if a["image_id"] == img_info["id"]), None)
        if ann is None:
            continue
        image = cv2.imread(str(image_root / img_info["file_name"]))
        if image is None:
            continue

        preds = model.predict(image)
        if not preds:
            oks50.append(0)
            oks75.append(0)
            continue

        pred_kps = [(kp.x, kp.y) for kp in preds[0].keypoints]
        bbox = ann.get("bbox") or compute_bbox_from_keypoints(ann["keypoints"])
        if bbox is None:
            continue

        oks50.append(paper_oks_per_image(ann["keypoints"], pred_kps, bbox, 0.5))
        oks75.append(paper_oks_per_image(ann["keypoints"], pred_kps, bbox, 0.75))

    return {
        "OKS@50": float(np.mean(oks50)) if oks50 else 0,
        "OKS@75": float(np.mean(oks75)) if oks75 else 0,
        "n": len(oks50),
    }


def ensure_bbox(gt_data):
    """Ensure all annotations have bbox computed from keypoints."""
    for ann in gt_data["annotations"]:
        bbox = compute_bbox_from_keypoints(ann["keypoints"])
        if bbox:
            ann["bbox"] = bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="Model names to evaluate")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--dataset", choices=["bp", "mydata", "both"], default="bp")
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    models_to_eval = args.models or list(ALL_MODELS.keys())

    # Load datasets
    datasets = {}
    if args.dataset in ("bp", "both"):
        with open(BP_TEST) as f:
            bp_data = json.load(f)
        ensure_bbox(bp_data)
        datasets["BP test (487)"] = (bp_data, BP_IMAGE_ROOT)

    if args.dataset in ("mydata", "both"):
        with open(MYDATA_GT) as f:
            my_data = json.load(f)
        ensure_bbox(my_data)
        datasets["mydata (18)"] = (my_data, MYDATA_ROOT)

    # Header
    print("=" * 80)
    print("Benchmark: Bizarre Pose paper-compatible OKS evaluation")
    print("=" * 80)
    print()
    print("Paper reference (Chen & Zwicker, WACV 2022):")
    for name, scores in PAPER_SCORES.items():
        print(f"  {name:<30} OKS@50={scores['OKS@50']:.3f}  OKS@75={scores['OKS@75']:.3f}")
    print()

    # Evaluate
    header_parts = [f"{'Model':<30}"]
    for ds_name in datasets:
        header_parts.append(f" {ds_name} OKS@50  OKS@75")
    print("  ".join(header_parts))
    print("-" * 80)

    for model_name in models_to_eval:
        try:
            cfg, ckpt = get_model(model_name)
        except KeyError as e:
            print(f"{model_name:<30} ERROR: {e}")
            continue

        try:
            model = RTMPoseEstimator(config=cfg, checkpoint=ckpt, device="cpu")
        except Exception as e:
            print(f"{model_name:<30} LOAD ERROR: {e}")
            continue

        row = f"{model_name:<30}"
        for ds_name, (gt_data, image_root) in datasets.items():
            result = evaluate_on_dataset(model, gt_data, image_root)
            row += f"  {result['OKS@50']:>13.3f}  {result['OKS@75']:>6.3f}"
        print(row)


if __name__ == "__main__":
    main()
