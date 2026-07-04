"""Convert generated sketch images + render GT into a COCO17 train json (R2b).

  python synth_to_coco.py --gen r2a/gen --renders r2a/renders_v3 \
      --out r2b/synth_train.json --prefix synth/ \
      [--fidelity r2a/fidelity_v3.json --min-pck02 0.75]

- keypoints come from the render gt.json (COCO17 order, v=0/1/2) — pixel-exact.
- bbox from visible keypoints + 15% margin (same convention as measure_fidelity).
- file_name = <prefix><scene>/<cond>.png so the training data_root can symlink
  the gen dir (e.g. data/merged/images/synth -> experiments/synth/r2a/gen).
- --fidelity: quality gate — drop samples whose measured PCK@0.2 (S2
  re-estimation, from measure_fidelity output) is below --min-pck02.
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="synth/")
    ap.add_argument("--fidelity", default="")
    ap.add_argument("--min-pck02", type=float, default=0.75)
    args = ap.parse_args()

    gate = None
    if args.fidelity:
        rows = json.load(open(args.fidelity))["rows"]
        gate = {(r["scene"], r["cond"]): r["pck02"] for r in rows}

    images, anns = [], []
    n_gated = n_nogate = 0
    img_id = ann_id = 1
    for scene in sorted(os.listdir(args.gen)):
        gt_path = os.path.join(args.renders, scene, "gt.json")
        sdir = os.path.join(args.gen, scene)
        if not os.path.isdir(sdir) or not os.path.exists(gt_path):
            continue
        gt = json.load(open(gt_path))
        kps = gt["keypoints"]  # [[x,y,v] x17]
        vis = [k for k in kps if k[2] > 0]
        if len(vis) < 4:
            continue
        xs, ys = [k[0] for k in vis], [k[1] for k in vis]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        mx, my = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
        bx = max(0.0, x0 - mx)
        by = max(0.0, y0 - my)
        bw = min(gt["img_w"], x1 + mx) - bx
        bh = min(gt["img_h"], y1 + my) - by
        flat = [round(v, 1) for k in kps for v in k]
        for f in sorted(os.listdir(sdir)):
            if not f.endswith(".png"):
                continue
            cond = f[:-4]
            if gate is not None:
                p = gate.get((scene, cond))
                if p is None:
                    n_nogate += 1
                    continue
                if p < args.min_pck02:
                    n_gated += 1
                    continue
            images.append({"id": img_id, "file_name": f"{args.prefix}{scene}/{f}",
                           "width": gt["img_w"], "height": gt["img_h"]})
            anns.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                         "keypoints": flat, "num_keypoints": len(vis),
                         "bbox": [round(v, 1) for v in (bx, by, bw, bh)],
                         "area": round(bw * bh, 1), "iscrowd": 0})
            img_id += 1
            ann_id += 1

    coco = {
        "info": {"description": "synthetic 3D->sketch pose data (R2b)"},
        "categories": [{"id": 1, "name": "person",
                        "keypoints": ["nose", "left_eye", "right_eye", "left_ear",
                                       "right_ear", "left_shoulder", "right_shoulder",
                                       "left_elbow", "right_elbow", "left_wrist",
                                       "right_wrist", "left_hip", "right_hip",
                                       "left_knee", "right_knee", "left_ankle",
                                       "right_ankle"],
                        "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                                      [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
                                      [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
                                      [4, 6], [5, 7]]}],
        "images": images, "annotations": anns,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(coco, f)
    print(f"wrote {args.out}: {len(images)} images "
          f"(gated out: {n_gated}, no fidelity row: {n_nogate})")


if __name__ == "__main__":
    main()
