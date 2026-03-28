"""Compare two models on mydata with 2-panel comparison images.

Usage:
    python -m scripts.eval.compare --left "Stage A" --right "Curriculum S2"
    python -m scripts.eval.compare --list
"""

from __future__ import annotations

import argparse
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

SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
    (200, 100, 50), (50, 100, 200), (100, 200, 50), (200, 50, 100),
    (50, 200, 100),
]

from scripts.eval.model_registry import ALL_MODELS, get_model, list_models
STAGES = ALL_MODELS


def draw_skeleton(image, keypoints, label, conf_threshold=0.3):
    vis = image.copy()
    h, w = vis.shape[:2]
    r = max(3, w // 150)
    high = 0

    for kp in keypoints:
        if kp.confidence > conf_threshold:
            high += 1
            cv2.circle(vis, (int(kp.x), int(kp.y)), r, (0, 255, 0), -1)
        elif kp.confidence > 0.1:
            cv2.circle(vis, (int(kp.x), int(kp.y)), max(2, w // 200), (0, 128, 255), -1)

    for i, j in SKELETON_PAIRS:
        if i < len(keypoints) and j < len(keypoints):
            k1, k2 = keypoints[i], keypoints[j]
            if k1.confidence > conf_threshold and k2.confidence > conf_threshold:
                cv2.line(vis, (int(k1.x), int(k1.y)), (int(k2.x), int(k2.y)),
                         COLORS[i % len(COLORS)], max(2, w // 300))

    cv2.putText(vis, f"{label}: {high}/17", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return vis


def main():
    parser = argparse.ArgumentParser(description="Compare two stages on mydata")
    parser.add_argument("--left", required=True, help="Left model name (e.g. 'Stage A')")
    parser.add_argument("--right", required=True, help="Right model name (e.g. 'Stage D')")
    parser.add_argument("--mydata", type=Path, default=Path("mydata"))
    parser.add_argument("--height", type=int, default=600, help="Output image height")
    args = parser.parse_args()

    if args.left not in STAGES:
        print(f"Unknown stage: {args.left}. Available: {list(STAGES.keys())}")
        sys.exit(1)
    if args.right not in STAGES:
        print(f"Unknown stage: {args.right}. Available: {list(STAGES.keys())}")
        sys.exit(1)

    # Output directory
    left_short = args.left.replace(" ", "").lower()
    right_short = args.right.replace(" ", "").lower()
    out_dir = Path(f"experiments/eval/comparisons/{left_short}_vs_{right_short}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print(f"Loading {args.left}...")
    cfg_l, ckpt_l = get_model(args.left)
    model_l = RTMPoseEstimator(config=cfg_l, checkpoint=ckpt_l, device="cpu")

    print(f"Loading {args.right}...")
    cfg_r, ckpt_r = get_model(args.right)
    model_r = RTMPoseEstimator(config=cfg_r, checkpoint=ckpt_r, device="cpu")

    # Process images
    for cat in ["chibi", "lineart", "part"]:
        cat_dir = args.mydata / cat
        if not cat_dir.exists():
            continue
        for p in sorted(list(cat_dir.glob("*.jpeg")) + list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg"))):
            img = cv2.imread(str(p))
            if img is None:
                continue

            pred_l = model_l.predict(img)
            pred_r = model_r.predict(img)

            vis_l = draw_skeleton(img, pred_l[0].keypoints, args.left) if pred_l else img.copy()
            vis_r = draw_skeleton(img, pred_r[0].keypoints, args.right) if pred_r else img.copy()

            scale = args.height / img.shape[0]
            tw = int(img.shape[1] * scale)
            comp = np.hstack([cv2.resize(vis_l, (tw, args.height)),
                              cv2.resize(vis_r, (tw, args.height))])

            out_path = out_dir / f"{cat}_{p.stem}_compare.jpg"
            cv2.imwrite(str(out_path), comp)
            print(f"  {cat}/{p.name}")

    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
