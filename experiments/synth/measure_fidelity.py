"""R2a label-fidelity gate: re-estimate keypoints on generated sketches with an
existing model and compare against the 3D-render GT that conditioned them.

Run with the PROJECT venv (needs mmpose):
  .venv/bin/python experiments/synth/measure_fidelity.py \
      --gen experiments/synth/r2a/gen --renders experiments/synth/r2a/renders_v2 \
      --model "Curriculum S2" --out experiments/synth/r2a/fidelity.json

Metric: PCK@0.1 (bbox-normalized) over GT keypoints with v>0, per condition.
Note: the checker model is itself weak on sketch styles, so absolute values
are a LOWER BOUND on true fidelity; use for relative comparison across
conditions + verify by eye with the comparison sheet.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import torch  # noqa: F401  (patch torch.load for own trusted checkpoints, same as scripts/eval/*)
_orig_load = torch.load


def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)


torch.load = _patched_load

import cv2
import numpy as np

sys.path.insert(0, os.getcwd())
from pose_estimation.core.types import BBox  # noqa: E402
from pose_estimation.models.rtmpose_estimator import RTMPoseEstimator  # noqa: E402
from scripts.eval.model_registry import get_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--model", default="Curriculum S2")
    ap.add_argument("--out", default="")
    ap.add_argument("--also-line", action="store_true",
                    help="also measure on the raw line render (procedural control group)")
    args = ap.parse_args()

    cfg, ckpt = get_model(args.model)
    est = RTMPoseEstimator(config=cfg, checkpoint=ckpt)

    rows = []
    scenes = sorted(os.listdir(args.gen))
    for scene in scenes:
        gt_path = os.path.join(args.renders, scene, "gt.json")
        if not os.path.exists(gt_path):
            print("no gt for", scene)
            continue
        gt = json.load(open(gt_path))
        kps = np.array(gt["keypoints"], dtype=np.float32)  # (17,3) x,y,v
        vis = kps[:, 2] > 0
        if vis.sum() < 4:
            continue
        xy = kps[vis, :2]
        x0, y0 = xy.min(0)
        x1, y1 = xy.max(0)
        mx, my = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
        bbox = BBox(x=max(0, x0 - mx), y=max(0, y0 - my),
                    w=min(gt["img_w"], x1 + mx) - max(0, x0 - mx),
                    h=min(gt["img_h"], y1 + my) - max(0, y0 - my))
        norm = max(bbox.w, bbox.h)

        targets = [os.path.join(args.gen, scene, f)
                   for f in sorted(os.listdir(os.path.join(args.gen, scene)))
                   if f.endswith(".png")]
        if args.also_line:
            targets.append(os.path.join(args.renders, scene, "line.png"))
        for fp in targets:
            img = cv2.imread(fp)
            if img is None:
                continue
            res = est.predict(img, bboxes=[bbox])
            if not res:
                continue
            pred = np.array([[k.x, k.y] for k in res[0].keypoints], dtype=np.float32)
            d = np.linalg.norm(pred - kps[:, :2], axis=1) / norm
            cond = os.path.basename(fp).replace(".png", "")
            row = {"scene": scene, "cond": cond,
                   "build": gt["build"], "pose": gt["pose"], "cam": gt["cam"],
                   "pck01": float((d[vis] < 0.10).mean()),
                   "pck02": float((d[vis] < 0.20).mean()),
                   "mean_nd": float(d[vis].mean())}
            rows.append(row)
            print(f"{scene} {cond} pck@0.1={row['pck01']:.2f}")

    agg = defaultdict(list)
    for r in rows:
        agg[r["cond"]].append(r)
        agg[f"build:{r['build']}|{r['cond']}"].append(r)
    summary = {k: {"n": len(v),
                   "pck01": round(float(np.mean([r["pck01"] for r in v])), 3),
                   "pck02": round(float(np.mean([r["pck02"] for r in v])), 3),
                   "mean_nd": round(float(np.mean([r["mean_nd"] for r in v])), 3)}
               for k, v in sorted(agg.items())}
    print(json.dumps({k: v for k, v in summary.items() if "|" not in k}, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "rows": rows, "summary": summary}, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
