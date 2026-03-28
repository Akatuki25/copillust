"""Convert Bizarre Pose dataset from custom 25-joint format to COCO17 JSON.

Bizarre Pose uses a custom format with 25 named keypoints stored as a dict.
This converter maps the 17 COCO keypoints and generates standard COCO JSON.

Joint mapping (Bizarre Pose → COCO17):
    nose         → 0: nose
    eye_left     → 1: left_eye
    eye_right    → 2: right_eye
    ear_left     → 3: left_ear
    ear_right    → 4: right_ear
    shoulder_left  → 5: left_shoulder
    shoulder_right → 6: right_shoulder
    elbow_left   → 7: left_elbow
    elbow_right  → 8: right_elbow
    wrist_left   → 9: left_wrist
    wrist_right  → 10: right_wrist
    hip_left     → 11: left_hip
    hip_right    → 12: right_hip
    knee_left    → 13: left_knee
    knee_right   → 14: right_knee
    ankle_left   → 15: left_ankle
    ankle_right  → 16: right_ankle

Usage:
    python -m pose_estimation.data.converters.bizarre_pose \\
        --data-root ./data/bizarre_pose/raw/bizarre_pose_dataset \\
        --output-dir ./data/bizarre_pose/coco
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pose_estimation.core.constants import NUM_KEYPOINTS
from pose_estimation.data.coco_utils import make_coco_category, save_coco_json

# Bizarre Pose joint name → COCO17 index.
BP_TO_COCO17: dict[str, int] = {
    "nose": 0,
    "eye_left": 1,
    "eye_right": 2,
    "ear_left": 3,
    "ear_right": 4,
    "shoulder_left": 5,
    "shoulder_right": 6,
    "elbow_left": 7,
    "elbow_right": 8,
    "wrist_left": 9,
    "wrist_right": 10,
    "hip_left": 11,
    "hip_right": 12,
    "knee_left": 13,
    "knee_right": 14,
    "ankle_left": 15,
    "ankle_right": 16,
}


def load_split_ids(filters_dir: Path, split: str) -> set[str]:
    """Load image IDs for a given split from CSV."""
    csv_path = filters_dir / f"accountably_{split}.csv"
    if not csv_path.exists():
        return set()
    with open(csv_path) as f:
        return {row[0] for row in csv.reader(f)}


def convert_bizarre_pose(
    data_root: Path,
    output_dir: Path,
    image_prefix: str = "images",
) -> dict[str, int]:
    """Convert Bizarre Pose annotations to COCO17 JSON per split.

    Args:
        data_root: Root of extracted bizarre_pose_dataset.
        output_dir: Where to write the COCO JSONs.
        image_prefix: Prefix for image file_name field.

    Returns:
        Stats dict with per-split counts.
    """
    ann_path = data_root / "raw" / "annotations.json"
    filters_dir = data_root / "_filters"
    images_dir = data_root / "raw" / "images"

    with open(ann_path) as f:
        raw_data: dict[str, Any] = json.load(f)

    # Load split assignments.
    splits = {}
    for split in ("train", "val", "test"):
        splits[split] = load_split_ids(filters_dir, split)

    output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}

    for split, split_ids in splits.items():
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        ann_id = 0

        for img_id_str in sorted(split_ids):
            entry = raw_data.get(img_id_str)
            if entry is None:
                continue

            img_id = int(img_id_str)
            bp_kps = entry.get("keypoints", {})
            size = entry.get("size", [0, 0])  # [width, height]
            bbox_raw = entry.get("bbox", [[0, 0], [0, 0]])  # [[x1,y1],[x2,y2]]

            # Build COCO keypoints.
            keypoints = [0.0] * (NUM_KEYPOINTS * 3)
            num_visible = 0
            for bp_name, coco_idx in BP_TO_COCO17.items():
                coord = bp_kps.get(bp_name)
                if coord and len(coord) == 2:
                    # Bizarre Pose stores [y, x] — verify
                    # Actually checking: first entry nose=[147,217], size=[600,400]
                    # Image is 600w x 400h. 147 < 400, 217 < 600.
                    # So it's [y, x] format.
                    y, x = coord[0], coord[1]
                    keypoints[coco_idx * 3] = float(x)
                    keypoints[coco_idx * 3 + 1] = float(y)
                    keypoints[coco_idx * 3 + 2] = 2  # visible
                    num_visible += 1

            if num_visible == 0:
                continue

            # Bbox: convert [[y1,x1],[y2,x2]] → [x, y, w, h].
            y1, x1 = bbox_raw[0]
            y2, x2 = bbox_raw[1]
            bx = float(min(x1, x2))
            by = float(min(y1, y2))
            bw = float(abs(x2 - x1))
            bh = float(abs(y2 - y1))

            # Image info.
            img_w, img_h = size[0], size[1]
            file_name = f"{image_prefix}/{img_id_str}.png"

            images.append({
                "id": img_id,
                "file_name": file_name,
                "width": img_w,
                "height": img_h,
            })

            ann_id += 1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [bx, by, bw, bh],
                "area": bw * bh,
                "num_keypoints": num_visible,
                "keypoints": keypoints,
                "iscrowd": 0,
                "render_type": "illustration",
            })

        coco_data = {
            "images": images,
            "annotations": annotations,
            "categories": [make_coco_category()],
        }

        out_path = output_dir / f"{split}.json"
        save_coco_json(coco_data, out_path)
        stats[split] = len(annotations)
        print(f"  {split}: {len(images)} images, {len(annotations)} annotations → {out_path}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Bizarre Pose to COCO17 format")
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="Root of bizarre_pose_dataset (contains raw/ and _filters/)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for COCO JSONs",
    )
    parser.add_argument(
        "--image-prefix", default="images",
        help="Prefix for image file_name",
    )
    args = parser.parse_args()

    print(f"Converting Bizarre Pose at {args.data_root}\n")
    stats = convert_bizarre_pose(args.data_root, args.output_dir, args.image_prefix)
    print(f"\nTotal: {sum(stats.values())} annotations")


if __name__ == "__main__":
    main()
