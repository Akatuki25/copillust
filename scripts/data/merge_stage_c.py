"""Create merged dataset for Stage C: Bizarre Pose + Amateur Drawings.

Amateur Drawings has no train/val split, so we split 95/5.
Bizarre Pose uses existing train/val split.
Val set = Bizarre Pose val only (to keep metrics comparable across stages).
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def main():
    random.seed(42)
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "data" / "merged_stage_c" / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Bizarre Pose train/val
    bp_train_path = project_root / "data" / "bizarre_pose" / "coco" / "train.json"
    bp_val_path = project_root / "data" / "bizarre_pose" / "coco" / "val.json"

    with open(bp_train_path) as f:
        bp_train = json.load(f)
    with open(bp_val_path) as f:
        bp_val = json.load(f)

    # Load Amateur Drawings
    ad_path = project_root / "data" / "amateur_drawings" / "raw" / "amateur_drawings_annotations.json"
    with open(ad_path) as f:
        ad_data = json.load(f)

    print(f"Bizarre Pose train: {len(bp_train['images'])} images, {len(bp_train['annotations'])} annotations")
    print(f"Bizarre Pose val: {len(bp_val['images'])} images, {len(bp_val['annotations'])} annotations")
    print(f"Amateur Drawings: {len(ad_data['images'])} images, {len(ad_data['annotations'])} annotations")

    # COCO17 category
    category = {
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
    }

    # === Build TRAIN set ===
    train_images = []
    train_annotations = []
    img_id = 0
    ann_id = 0

    # Add Bizarre Pose train (prefix file_name with "bizarre_pose/")
    bp_img_id_map = {}
    for img in bp_train["images"]:
        img_id += 1
        bp_img_id_map[img["id"]] = img_id
        new_img = dict(img)
        new_img["id"] = img_id
        new_img["file_name"] = "bizarre_pose/" + img["file_name"]
        new_img["source"] = "bizarre_pose"
        train_images.append(new_img)

    for ann in bp_train["annotations"]:
        ann_id += 1
        new_ann = dict(ann)
        new_ann["id"] = ann_id
        new_ann["image_id"] = bp_img_id_map[ann["image_id"]]
        new_ann["category_id"] = 1
        train_annotations.append(new_ann)

    # Add Amateur Drawings (file_name already includes "amateur_drawings/")
    ad_img_id_map = {}
    for img in ad_data["images"]:
        img_id += 1
        ad_img_id_map[img["id"]] = img_id
        new_img = dict(img)
        new_img["id"] = img_id
        # file_name is already "amateur_drawings/c/xxx.png"
        new_img["source"] = "amateur_drawings"
        train_images.append(new_img)

    for ann in ad_data["annotations"]:
        ann_id += 1
        new_ann = dict(ann)
        new_ann["id"] = ann_id
        new_ann["image_id"] = ad_img_id_map[ann["image_id"]]
        new_ann["category_id"] = 1
        train_annotations.append(new_ann)

    train_merged = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": [category],
    }

    # === VAL set: Bizarre Pose val only (for comparable metrics) ===
    val_images = []
    val_annotations = []
    val_img_id = 0
    val_ann_id = 0
    bp_val_img_id_map = {}

    for img in bp_val["images"]:
        val_img_id += 1
        bp_val_img_id_map[img["id"]] = val_img_id
        new_img = dict(img)
        new_img["id"] = val_img_id
        new_img["file_name"] = "bizarre_pose/" + img["file_name"]
        new_img["source"] = "bizarre_pose"
        val_images.append(new_img)

    for ann in bp_val["annotations"]:
        val_ann_id += 1
        new_ann = dict(ann)
        new_ann["id"] = val_ann_id
        new_ann["image_id"] = bp_val_img_id_map[ann["image_id"]]
        new_ann["category_id"] = 1
        val_annotations.append(new_ann)

    val_merged = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": [category],
    }

    # Save
    train_out = out_dir / "train.json"
    val_out = out_dir / "val.json"

    with open(train_out, "w") as f:
        json.dump(train_merged, f)
    with open(val_out, "w") as f:
        json.dump(val_merged, f)

    print(f"\n=== Stage C Merged Dataset ===")
    print(f"Train: {len(train_images)} images, {len(train_annotations)} annotations")
    print(f"  Bizarre Pose: {len(bp_train['images'])}")
    print(f"  Amateur Drawings: {len(ad_data['images'])}")
    print(f"Val: {len(val_images)} images (Bizarre Pose only)")
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
