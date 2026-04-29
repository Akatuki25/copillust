"""Fixed evaluator for autoexp experiments.

Evaluates on:
1. Bizarre Pose test set (487 images) — primary benchmark
2. mydata (18 GT-annotated illustrations) — sanity check

Primary metric: bp_oks75 (OKS@75).
Baseline (Curriculum S2): 0.801

Usage:
    from autoexp.evaluate import run_evaluation
    result = run_evaluation("path/to/config.py", "path/to/checkpoint.pth")
    print(result["bp_oks75"])  # 0.805

    # CLI
    python -m autoexp.evaluate \\
        --config pose_estimation/models/configs/... \\
        --checkpoint experiments/train/.../best.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BP_TEST = PROJECT_ROOT / "data/bizarre_pose/coco/test.json"
BP_IMAGE_ROOT = PROJECT_ROOT / "data/bizarre_pose/raw/bizarre_pose_dataset/raw"
MYDATA_GT = PROJECT_ROOT / "mydata/annotations.json"
MYDATA_ROOT = PROJECT_ROOT / "mydata"

# If mydata OKS@75 drops below this, it's a sanity failure.
MYDATA_SANITY_THRESHOLD = 0.40


def _compute_bbox_from_keypoints(keypoints_flat: list[float]) -> list[float] | None:
    """Compute bbox [x,y,w,h] from COCO17 flat keypoints."""
    kps = np.array(keypoints_flat).reshape(17, 3)
    visible = kps[kps[:, 2] > 0]
    if len(visible) == 0:
        return None
    x1, y1 = visible[:, 0].min(), visible[:, 1].min()
    x2, y2 = visible[:, 0].max(), visible[:, 1].max()
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def _paper_oks(gt_flat: list[float], pred_xy: list[tuple[float, float]],
               bbox: list[float], threshold: float) -> int:
    """Paper-compatible per-image OKS correctness (0 or 1).

    Uses COCO sigmas. Returns 1 if OKS >= threshold, else 0.
    """
    SIGMAS = np.array([
        0.026, 0.025, 0.025, 0.035, 0.035,
        0.079, 0.079, 0.072, 0.072, 0.062, 0.062,
        0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
    ])
    gt = np.array(gt_flat).reshape(17, 3)
    area = (bbox[2] * bbox[3]) or 1.0

    valid = gt[:, 2] > 0
    if not valid.any():
        return 0

    pred = np.array(pred_xy)  # (17, 2)
    dx = pred[:, 0] - gt[:, 0]
    dy = pred[:, 1] - gt[:, 1]
    s = np.sqrt(area)
    e = (dx**2 + dy**2) / (2 * (s * SIGMAS)**2)
    oks = np.exp(-e)[valid].mean()
    return int(oks >= threshold)


def _load_model(config_path: str, checkpoint_path: str, device: str = "cpu"):
    """Load RTMPoseEstimator."""
    import torch
    _orig = torch.load
    def _patched(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig(*a, **kw)
    torch.load = _patched

    from pose_estimation.models.rtmpose_estimator import RTMPoseEstimator
    return RTMPoseEstimator(config=config_path, checkpoint=checkpoint_path, device=device)


def _eval_dataset(
    model,
    gt_data: dict,
    image_root: Path,
) -> dict[str, float | int]:
    """Evaluate model on one dataset."""
    import cv2
    oks50, oks75 = [], []

    for img_info in gt_data["images"]:
        ann = next(
            (a for a in gt_data["annotations"] if a["image_id"] == img_info["id"]),
            None,
        )
        if ann is None:
            continue

        img_path = image_root / img_info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        preds = model.predict(image)
        if not preds:
            oks50.append(0)
            oks75.append(0)
            continue

        pred_xy = [(kp.x, kp.y) for kp in preds[0].keypoints]
        bbox = ann.get("bbox") or _compute_bbox_from_keypoints(ann["keypoints"])
        if bbox is None:
            continue

        oks50.append(_paper_oks(ann["keypoints"], pred_xy, bbox, 0.50))
        oks75.append(_paper_oks(ann["keypoints"], pred_xy, bbox, 0.75))

    n = len(oks50)
    return {
        "oks50": float(np.mean(oks50)) if oks50 else 0.0,
        "oks75": float(np.mean(oks75)) if oks75 else 0.0,
        "n": n,
    }


def run_evaluation(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str = "cpu",
    skip_mydata: bool = False,
) -> dict:
    """Run the fixed evaluation protocol.

    Args:
        config_path: MMPose config file path.
        checkpoint_path: Model checkpoint path.
        device: "cpu", "cuda", or "mps".
        skip_mydata: Skip mydata sanity check (faster, less informative).

    Returns:
        dict with:
            bp_oks50, bp_oks75, bp_n         — Bizarre Pose test results
            mydata_oks50, mydata_oks75, mydata_n — mydata results (0 if skipped)
            sanity_ok                          — True if mydata OKS@75 >= threshold
    """
    model = _load_model(str(config_path), str(checkpoint_path), device)

    with open(BP_TEST) as f:
        bp_data = json.load(f)
    # Ensure all annotations have bbox
    for ann in bp_data["annotations"]:
        if not ann.get("bbox"):
            ann["bbox"] = _compute_bbox_from_keypoints(ann["keypoints"])

    bp_result = _eval_dataset(model, bp_data, BP_IMAGE_ROOT)

    mydata_result = {"oks50": 0.0, "oks75": 0.0, "n": 0}
    if not skip_mydata and MYDATA_GT.exists():
        with open(MYDATA_GT) as f:
            my_data = json.load(f)
        for ann in my_data["annotations"]:
            if not ann.get("bbox"):
                ann["bbox"] = _compute_bbox_from_keypoints(ann["keypoints"])
        mydata_result = _eval_dataset(model, my_data, MYDATA_ROOT)

    sanity_ok = (
        mydata_result["oks75"] >= MYDATA_SANITY_THRESHOLD
        or mydata_result["n"] == 0
    )

    return {
        "bp_oks50": bp_result["oks50"],
        "bp_oks75": bp_result["oks75"],
        "bp_n": bp_result["n"],
        "mydata_oks50": mydata_result["oks50"],
        "mydata_oks75": mydata_result["oks75"],
        "mydata_n": mydata_result["n"],
        "sanity_ok": sanity_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed evaluator for autoexp")
    parser.add_argument("--config", required=True, help="MMPose config path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--skip-mydata", action="store_true")
    args = parser.parse_args()

    result = run_evaluation(args.config, args.checkpoint, args.device, args.skip_mydata)

    print("\nEvaluation Results")
    print("=" * 50)
    print(f"Bizarre Pose test ({result['bp_n']} images):")
    print(f"  OKS@50: {result['bp_oks50']:.3f}")
    print(f"  OKS@75: {result['bp_oks75']:.3f}  ← primary metric (baseline: 0.801)")

    if result["mydata_n"] > 0:
        print(f"\nmydata sanity ({result['mydata_n']} images):")
        print(f"  OKS@50: {result['mydata_oks50']:.3f}")
        print(f"  OKS@75: {result['mydata_oks75']:.3f}")
        print(f"  Sanity: {'PASS' if result['sanity_ok'] else 'FAIL (degraded!)'}")

    delta = result["bp_oks75"] - 0.801
    sign = "+" if delta >= 0 else ""
    print(f"\nDelta vs baseline: {sign}{delta:.3f}")


if __name__ == "__main__":
    main()
