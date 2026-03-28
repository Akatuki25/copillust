"""Merge multiple COCO-format datasets with configurable sampling ratios.

Handles:
- ID renumbering to avoid collisions
- Image path prefixing per dataset
- Oversampling smaller datasets to achieve target ratios
- Source tracking in metadata

Usage:
    python -m pose_estimation.data.merge \\
        --config merge_config.json \\
        --output ./data/merged/annotations/train.json

Or programmatically via merge_datasets().
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from pose_estimation.data.coco_utils import load_coco_json, make_coco_category, save_coco_json


def merge_datasets(
    sources: list[dict[str, Any]],
    output_path: Path,
    seed: int = 42,
) -> dict[str, int]:
    """Merge multiple COCO datasets into one.

    Args:
        sources: List of source configs, each with:
            - "annotation_file": path to COCO JSON
            - "image_prefix": prefix to prepend to file_name (e.g., "HumanArt/")
            - "ratio": target sampling ratio (0.0–1.0)
            - "source_name": identifier for tracking
        output_path: Where to write the merged JSON.
        seed: Random seed for reproducible sampling.

    Returns:
        Stats dict with per-source counts.
    """
    random.seed(seed)

    # Load all sources.
    loaded: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for src in sources:
        data = load_coco_json(src["annotation_file"])
        loaded.append((src, data))

    # Compute target counts based on ratios.
    # Find the source that, at its natural size, constrains the total.
    total_natural = sum(len(d["annotations"]) for _, d in loaded)
    ratio_sum = sum(s["ratio"] for s in sources)

    # Normalize ratios.
    for src in sources:
        src["_norm_ratio"] = src["ratio"] / ratio_sum

    # Determine how many annotations to take from each source.
    # Strategy: oversample smaller datasets to match the ratio.
    target_total = total_natural  # Keep total roughly the same.
    source_targets: list[int] = []
    for src, data in loaded:
        target_n = int(target_total * src["_norm_ratio"])
        source_targets.append(target_n)

    # Merge.
    merged_images: list[dict[str, Any]] = []
    merged_annotations: list[dict[str, Any]] = []
    image_id_offset = 0
    ann_id_offset = 0
    stats: dict[str, int] = {}

    for (src, data), target_n in zip(loaded, source_targets):
        name = src.get("source_name", "unknown")
        prefix = src.get("image_prefix", "")
        annotations = data.get("annotations", [])
        images = data.get("images", [])

        # Build image_id → image dict lookup.
        img_lookup = {img["id"]: img for img in images}

        # Sample or oversample annotations.
        if len(annotations) >= target_n:
            sampled = random.sample(annotations, target_n)
        else:
            # Oversample: repeat full dataset + sample remainder.
            repeats = target_n // len(annotations)
            remainder = target_n % len(annotations)
            sampled = annotations * repeats + random.sample(annotations, remainder)

        # Collect unique image IDs from sampled annotations.
        sampled_img_ids = set()
        for ann in sampled:
            sampled_img_ids.add(ann["image_id"])

        # Renumber images.
        old_to_new_img_id: dict[int, int] = {}
        for old_id in sorted(sampled_img_ids):
            if old_id in old_to_new_img_id:
                continue
            new_id = image_id_offset + len(old_to_new_img_id) + 1
            old_to_new_img_id[old_id] = new_id

            img = img_lookup.get(old_id, {"id": old_id, "file_name": f"{old_id}.jpg"})
            new_img = dict(img)
            new_img["id"] = new_id
            fname = new_img.get("file_name", "")
            if prefix and not fname.startswith(prefix):
                new_img["file_name"] = f"{prefix}{fname}"
            new_img["source"] = name
            merged_images.append(new_img)

        # Renumber annotations.
        for ann in sampled:
            new_ann = dict(ann)
            ann_id_offset += 1
            new_ann["id"] = ann_id_offset
            new_ann["image_id"] = old_to_new_img_id[ann["image_id"]]
            new_ann["source"] = name
            merged_annotations.append(new_ann)

        image_id_offset += len(old_to_new_img_id)
        stats[name] = len(sampled)

    merged = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": [make_coco_category()],
    }

    save_coco_json(merged, output_path)

    print(f"Merged dataset: {len(merged_annotations)} annotations, {len(merged_images)} images")
    for name, count in stats.items():
        pct = count / len(merged_annotations) * 100 if merged_annotations else 0
        print(f"  {name}: {count} ({pct:.1f}%)")
    print(f"Output: {output_path}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge COCO datasets")
    parser.add_argument(
        "--config", type=Path, required=True,
        help="JSON config file with sources list",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    merge_datasets(config["sources"], args.output, args.seed)


if __name__ == "__main__":
    main()
