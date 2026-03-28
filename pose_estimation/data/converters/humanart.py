"""Human-Art dataset validator and path resolver.

Human-Art is already COCO17 compatible. This converter only:
1. Validates that the expected JSON files exist and are well-formed
2. Verifies keypoints are 17-point COCO format
3. Extracts scenario metadata from the `category` field

Usage:
    python -m pose_estimation.data.converters.humanart --data-root ./data/HumanArt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pose_estimation.core.constants import NUM_KEYPOINTS
from pose_estimation.data.coco_utils import load_coco_json, validate_coco_json


# Expected annotation files from the Human-Art repository.
EXPECTED_FILES = [
    "training_humanart_coco.json",
    "validation_humanart.json",
]

OPTIONAL_FILES = [
    "validation_humanart_cartoon.json",
    "training_humanart.json",
]


def find_annotation_files(data_root: Path) -> dict[str, Path]:
    """Locate Human-Art annotation JSON files.

    Searches recursively under data_root since the repo structure
    may nest them in subdirectories.
    """
    found: dict[str, Path] = {}
    for name in EXPECTED_FILES + OPTIONAL_FILES:
        matches = list(data_root.rglob(name))
        if matches:
            found[name] = matches[0]
    return found


def validate_humanart(data_root: Path) -> dict[str, list[str]]:
    """Validate Human-Art annotations.

    Returns:
        Dict mapping filename → list of warnings (empty list = OK).
    """
    results: dict[str, list[str]] = {}
    files = find_annotation_files(data_root)

    for name in EXPECTED_FILES:
        if name not in files:
            results[name] = [f"MISSING: {name} not found under {data_root}"]
            continue

        path = files[name]
        data = load_coco_json(path)
        warnings = validate_coco_json(data)

        # Verify COCO17 keypoint count on a sample.
        annotations = data.get("annotations", [])
        if annotations:
            sample = annotations[0]
            kps = sample.get("keypoints", [])
            if len(kps) != NUM_KEYPOINTS * 3:
                warnings.append(
                    f"First annotation has {len(kps)} keypoint values, "
                    f"expected {NUM_KEYPOINTS * 3} (COCO17)"
                )

        n_images = len(data.get("images", []))
        n_anns = len(annotations)
        print(f"  {name}: {n_images} images, {n_anns} annotations")

        # List unique categories/scenarios.
        categories = set()
        for img in data.get("images", []):
            cat = img.get("category")
            if cat:
                categories.add(cat)
        if categories:
            print(f"    Scenarios: {sorted(categories)}")

        results[name] = warnings

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Human-Art dataset")
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    print(f"Validating Human-Art at {args.data_root}\n")
    results = validate_humanart(args.data_root)

    all_ok = True
    for name, warnings in results.items():
        if warnings:
            all_ok = False
            print(f"\n  [WARN] {name}:")
            for w in warnings:
                print(f"    - {w}")

    if all_ok:
        print("\n  All checks passed. Human-Art is ready for use.")
    else:
        print("\n  Some issues found. See warnings above.")


if __name__ == "__main__":
    main()
