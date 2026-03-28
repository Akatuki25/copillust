"""Convert Amateur Drawings dataset from custom 16-joint format to COCO17 JSON.

Amateur Drawings (Meta/Facebook AnimatedDrawings) uses a custom hierarchical
joint format with 16 joints.  This converter:

1. Reads `amateur_drawings_annotations.json`
2. Maps 16 joints → COCO17 keypoints:
   - 12 joints map directly or approximately (hand→wrist, foot→ankle)
   - 5 face keypoints (nose, eyes, ears) are set to [0,0,0] (unlabeled)
3. Optionally filters to keep only a subset (e.g., lineart-like drawings)
4. Outputs a COCO-compatible JSON

Joint mapping:
    Amateur Drawings (16)    →  COCO17 (17)
    ─────────────────────────────────────────
    (none)                   →  0: nose          [0,0,0]
    (none)                   →  1: left_eye      [0,0,0]
    (none)                   →  2: right_eye     [0,0,0]
    (none)                   →  3: left_ear      [0,0,0]
    (none)                   →  4: right_ear     [0,0,0]
    left_shoulder            →  5: left_shoulder
    right_shoulder           →  6: right_shoulder
    left_elbow               →  7: left_elbow
    right_elbow              →  8: right_elbow
    left_hand                →  9: left_wrist    (approx)
    right_hand               → 10: right_wrist   (approx)
    left_hip                 → 11: left_hip
    right_hip                → 12: right_hip
    left_knee                → 13: left_knee
    right_knee               → 14: right_knee
    left_foot                → 15: left_ankle    (approx)
    right_foot               → 16: right_ankle   (approx)

Usage:
    python -m pose_estimation.data.converters.amateur_drawings \\
        --annotations ./data/amateur_drawings/raw/amateur_drawings_annotations.json \\
        --output ./data/amateur_drawings/converted/train.json \\
        --max-samples 30000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from pose_estimation.core.constants import NUM_KEYPOINTS
from pose_estimation.data.coco_utils import make_coco_category, save_coco_json

# Mapping from Amateur Drawings joint name → COCO17 keypoint index.
# Joints not in this map have no COCO17 equivalent (root, hip, torso, neck
# are structural joints that don't directly map to COCO keypoints).
AD_TO_COCO17: dict[str, int] = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_hand": 9,       # approximate: hand ≈ wrist
    "right_hand": 10,     # approximate: hand ≈ wrist
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_foot": 15,      # approximate: foot ≈ ankle
    "right_foot": 16,     # approximate: foot ≈ ankle
}

# Face indices that will be set to [0, 0, 0] (unlabeled).
FACE_INDICES = [0, 1, 2, 3, 4]


def convert_single_entry(
    entry: dict[str, Any],
    image_id: int,
    annotation_id: int,
    image_dir: str = "images",
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Convert a single Amateur Drawings entry to COCO image + annotation dicts.

    Args:
        entry: A single entry from amateur_drawings_annotations.json.
        image_id: COCO image ID to assign.
        annotation_id: COCO annotation ID to assign.
        image_dir: Prefix for the file_name field.

    Returns:
        (image_dict, annotation_dict) or None if entry is malformed.
    """
    # Extract joints.  The format varies:
    #   - Some entries have joints as a list of {"name": ..., "loc": [x, y]}
    #   - Some have a nested structure
    joints = entry.get("joints")
    if not joints:
        return None

    # Parse joints into a name → (x, y) mapping.
    joint_map: dict[str, tuple[float, float]] = {}
    if isinstance(joints, list):
        for j in joints:
            name = j.get("name", "")
            loc = j.get("loc", [0, 0])
            if name and len(loc) == 2:
                joint_map[name] = (float(loc[0]), float(loc[1]))
    elif isinstance(joints, dict):
        for name, loc in joints.items():
            if isinstance(loc, (list, tuple)) and len(loc) == 2:
                joint_map[name] = (float(loc[0]), float(loc[1]))

    if not joint_map:
        return None

    # Build COCO17 keypoints array: [x1, y1, v1, x2, y2, v2, ...].
    keypoints = [0.0] * (NUM_KEYPOINTS * 3)
    num_mapped = 0

    for ad_name, coco_idx in AD_TO_COCO17.items():
        if ad_name in joint_map:
            x, y = joint_map[ad_name]
            keypoints[coco_idx * 3] = x
            keypoints[coco_idx * 3 + 1] = y
            keypoints[coco_idx * 3 + 2] = 2  # visibility = labeled and visible
            num_mapped += 1

    # Face keypoints remain [0, 0, 0] (unlabeled) — already zero-initialized.

    if num_mapped == 0:
        return None

    # Bbox: use entry bbox if available, otherwise compute from keypoints.
    bbox = entry.get("bbox", None)
    if bbox and len(bbox) == 4:
        bx, by, bw, bh = [float(v) for v in bbox]
    else:
        # Compute from mapped keypoints.
        xs = [keypoints[i * 3] for i in range(NUM_KEYPOINTS) if keypoints[i * 3 + 2] > 0]
        ys = [keypoints[i * 3 + 1] for i in range(NUM_KEYPOINTS) if keypoints[i * 3 + 2] > 0]
        if not xs:
            return None
        bx, by = min(xs), min(ys)
        bw, bh = max(xs) - bx, max(ys) - by
        # Add padding.
        pad_x, pad_y = bw * 0.1, bh * 0.1
        bx -= pad_x
        by -= pad_y
        bw += 2 * pad_x
        bh += 2 * pad_y

    # Image dimensions.
    img_w = entry.get("width", int(bx + bw + 1))
    img_h = entry.get("height", int(by + bh + 1))

    # File name.
    file_name = entry.get("file_name", entry.get("image_path", f"{image_id:06d}.png"))
    if image_dir:
        file_name = f"{image_dir}/{file_name}"

    image_dict = {
        "id": image_id,
        "file_name": file_name,
        "width": img_w,
        "height": img_h,
    }

    annotation_dict = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [bx, by, bw, bh],
        "area": bw * bh,
        "num_keypoints": num_mapped,
        "keypoints": keypoints,
        "iscrowd": 0,
        "render_type": "drawing",
    }

    return image_dict, annotation_dict


def convert_amateur_drawings(
    annotations_path: Path,
    output_path: Path,
    max_samples: int | None = None,
    image_dir: str = "images",
) -> dict[str, int]:
    """Convert the full Amateur Drawings annotation file to COCO17 JSON.

    Args:
        annotations_path: Path to amateur_drawings_annotations.json.
        output_path: Where to write the converted COCO JSON.
        max_samples: Maximum number of entries to convert (None = all).
        image_dir: Prefix for image file paths.

    Returns:
        Stats dict with counts of processed, converted, and skipped entries.
    """
    print(f"Loading {annotations_path}...")
    with open(annotations_path) as f:
        raw_data = json.load(f)

    # The annotations file may be a list or a dict with entries.
    if isinstance(raw_data, list):
        entries = raw_data
    elif isinstance(raw_data, dict):
        # Try common keys.
        entries = raw_data.get("annotations", raw_data.get("data", []))
        if not entries and len(raw_data) == 1:
            entries = next(iter(raw_data.values()))
    else:
        raise ValueError(f"Unexpected top-level type: {type(raw_data)}")

    if max_samples:
        entries = entries[:max_samples]

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    skipped = 0

    for i, entry in enumerate(tqdm(entries, desc="Converting")):
        result = convert_single_entry(
            entry,
            image_id=i + 1,
            annotation_id=i + 1,
            image_dir=image_dir,
        )
        if result is None:
            skipped += 1
            continue
        img_dict, ann_dict = result
        images.append(img_dict)
        annotations.append(ann_dict)

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": [make_coco_category()],
    }

    save_coco_json(coco_data, output_path)

    stats = {
        "total_entries": len(entries),
        "converted": len(annotations),
        "skipped": skipped,
    }
    print(f"Converted: {stats['converted']}, Skipped: {stats['skipped']}")
    print(f"Output: {output_path}")
    print()
    print("NOTE: Face keypoints (nose, eyes, ears) are set to [0,0,0] (unlabeled).")
    print("      MMPose will ignore these in loss computation (visibility=0).")
    print("      hand→wrist and foot→ankle are approximate mappings.")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Amateur Drawings to COCO17 format"
    )
    parser.add_argument(
        "--annotations", type=Path, required=True,
        help="Path to amateur_drawings_annotations.json",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output COCO JSON path",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max entries to convert (default: all)",
    )
    parser.add_argument(
        "--image-dir", default="images",
        help="Prefix for image file_name field",
    )
    args = parser.parse_args()

    convert_amateur_drawings(
        args.annotations, args.output, args.max_samples, args.image_dir
    )


if __name__ == "__main__":
    main()
