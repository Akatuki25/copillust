"""Merge BP train annotations with synthetic COCO json into one train file (R2b).

  python merge_r2b.py --bp data/merged/annotations/train.json \
      --synth experiments/synth/r2b/synth_train.json \
      --out data/merged/annotations/train_r2b.json

Both inputs are COCO17; ids are re-numbered to avoid collisions. file_name
prefixes (bizarre_pose/, synth/) must already point under data_root images/.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp", required=True)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bp = json.load(open(args.bp))
    sy = json.load(open(args.synth))

    images, anns = [], []
    img_id = ann_id = 1
    for src in (bp, sy):
        remap = {}
        for im in src["images"]:
            remap[im["id"]] = img_id
            im = dict(im, id=img_id)
            images.append(im)
            img_id += 1
        for a in src["annotations"]:
            a = dict(a, id=ann_id, image_id=remap[a["image_id"]])
            anns.append(a)
            ann_id += 1

    out = {"info": {"description": "BP + synthetic mixed train (R2b)"},
           "categories": bp["categories"], "images": images, "annotations": anns}
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}: {len(bp['images'])} BP + {len(sy['images'])} synth "
          f"= {len(images)} images")


if __name__ == "__main__":
    main()
