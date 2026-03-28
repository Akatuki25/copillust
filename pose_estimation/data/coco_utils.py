"""Utilities for reading, writing, and validating COCO-format JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pose_estimation.core.constants import COCO17_KEYPOINTS, COCO17_SKELETON, NUM_KEYPOINTS
from pose_estimation.core.types import BBox, Keypoint, PoseResult


def load_coco_json(path: str | Path) -> dict[str, Any]:
    """Load a COCO-format JSON file."""
    with open(path) as f:
        return json.load(f)


def save_coco_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a COCO-format JSON file with compact formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def make_coco_category() -> dict[str, Any]:
    """Create the standard COCO person category entry."""
    # COCO skeleton uses 1-indexed keypoint pairs.
    skeleton_1indexed = [[a + 1, b + 1] for a, b in COCO17_SKELETON]
    return {
        "id": 1,
        "name": "person",
        "keypoints": COCO17_KEYPOINTS,
        "skeleton": skeleton_1indexed,
    }


def validate_coco_annotation(ann: dict[str, Any]) -> list[str]:
    """Check a single COCO annotation for common issues.

    Returns a list of warning messages (empty if all OK).
    """
    warnings: list[str] = []
    ann_id = ann.get("id", "?")

    kps = ann.get("keypoints", [])
    if len(kps) != NUM_KEYPOINTS * 3:
        warnings.append(
            f"ann {ann_id}: keypoints length {len(kps)}, expected {NUM_KEYPOINTS * 3}"
        )

    bbox = ann.get("bbox", [])
    if len(bbox) != 4:
        warnings.append(f"ann {ann_id}: bbox length {len(bbox)}, expected 4")
    elif bbox[2] <= 0 or bbox[3] <= 0:
        warnings.append(f"ann {ann_id}: bbox has non-positive width/height: {bbox}")

    return warnings


def validate_coco_json(data: dict[str, Any]) -> list[str]:
    """Validate the overall structure of a COCO JSON dataset.

    Returns a list of warning/error messages.
    """
    warnings: list[str] = []

    for section in ("images", "annotations", "categories"):
        if section not in data:
            warnings.append(f"missing top-level key: '{section}'")

    image_ids = {img["id"] for img in data.get("images", [])}
    for ann in data.get("annotations", []):
        if ann.get("image_id") not in image_ids:
            warnings.append(
                f"ann {ann.get('id')}: image_id {ann.get('image_id')} not found in images"
            )
        warnings.extend(validate_coco_annotation(ann))

    return warnings


def coco_annotation_to_pose_result(
    ann: dict[str, Any],
    image_meta: dict[str, Any] | None = None,
) -> PoseResult:
    """Convert a COCO annotation dict to a PoseResult.

    Args:
        ann: A single annotation from the COCO JSON ``annotations`` list.
        image_meta: Optional image-level metadata (render_type, etc.).
    """
    raw_kps = ann.get("keypoints", [0] * NUM_KEYPOINTS * 3)
    keypoints: list[Keypoint] = []
    for i in range(NUM_KEYPOINTS):
        x = float(raw_kps[i * 3])
        y = float(raw_kps[i * 3 + 1])
        v = int(raw_kps[i * 3 + 2])
        keypoints.append(Keypoint(x=x, y=y, visibility=v))

    bx, by, bw, bh = ann.get("bbox", [0, 0, 0, 0])
    bbox = BBox(x=float(bx), y=float(by), w=float(bw), h=float(bh))

    metadata: dict[str, Any] = {}
    for key in ("body_type", "frame_type", "render_type", "background_complexity",
                "view_type", "category"):
        val = ann.get(key) or (image_meta or {}).get(key)
        if val is not None:
            metadata[key] = val

    return PoseResult(
        keypoints=keypoints,
        bbox=bbox,
        score=float(ann.get("score", 1.0)),
        metadata=metadata,
    )


def pose_result_to_coco_annotation(
    pose: PoseResult,
    annotation_id: int,
    image_id: int,
    category_id: int = 1,
) -> dict[str, Any]:
    """Convert a PoseResult back to a COCO annotation dict."""
    kps = pose.to_coco_keypoints()
    num_visible = sum(1 for kp in pose.keypoints if kp.visibility > 0)
    ann: dict[str, Any] = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": list(pose.bbox.to_xywh()),
        "area": pose.bbox.area,
        "num_keypoints": num_visible,
        "keypoints": kps,
        "iscrowd": 0,
    }
    if pose.metadata:
        ann.update(pose.metadata)
    return ann
