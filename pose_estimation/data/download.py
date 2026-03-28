"""Download and organize public datasets for pose estimation training.

Usage:
    python -m pose_estimation.data.download --data-root ./data --dataset all
    python -m pose_estimation.data.download --data-root ./data --dataset humanart
    python -m pose_estimation.data.download --data-root ./data --dataset bizarre_pose
    python -m pose_estimation.data.download --data-root ./data --dataset amateur_drawings
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)


# ---------------------------------------------------------------------------
# Human-Art
# ---------------------------------------------------------------------------

def download_humanart(data_root: Path) -> None:
    """Clone Human-Art repo and provide instructions for image download.

    Human-Art images require registration via Google Form.
    The repo itself contains annotation JSON files.
    """
    dest = data_root / "HumanArt"
    if dest.exists():
        print(f"[Human-Art] Already exists at {dest}, skipping clone.")
    else:
        print("[Human-Art] Cloning IDEA-Research/HumanArt...")
        _run(["git", "clone", "--depth", "1",
              "https://github.com/IDEA-Research/HumanArt.git", str(dest)])

    print()
    print("[Human-Art] Annotations available:")
    for f in sorted(dest.rglob("*.json")):
        print(f"  {f.relative_to(dest)}")

    print()
    print("[Human-Art] IMAGE DOWNLOAD REQUIRED:")
    print("  Human-Art images require Google Form registration.")
    print("  1. Visit: https://github.com/IDEA-Research/HumanArt")
    print("  2. Fill out the Google Form linked in the README")
    print("  3. Download images and place them under:")
    print(f"     {dest}/")
    print()
    print("  Expected annotations (COCO17 compatible):")
    print("    - training_humanart_coco.json  (training, COCO format)")
    print("    - validation_humanart.json     (validation)")
    print("    - validation_humanart_cartoon.json (cartoon subset for test)")
    print()
    print("  Label content:")
    print("    - 17 COCO keypoints: [x,y,v] * 17 (v: 0=unlabeled, 1=occluded, 2=visible)")
    print("    - 21 extended keypoints in 'keypoints_21' field (optional)")
    print("    - bbox: [x, y, width, height]")
    print("    - category: scenario classification (e.g. 'cartoon', 'digital_art')")


# ---------------------------------------------------------------------------
# Bizarre Pose
# ---------------------------------------------------------------------------

def download_bizarre_pose(data_root: Path) -> None:
    """Clone bizarre-pose-estimator repo and provide dataset download instructions.

    The dataset ZIP must be downloaded separately following the repo instructions.
    """
    dest = data_root / "bizarre_pose"
    repo_dir = dest / "repo"

    if repo_dir.exists():
        print(f"[Bizarre Pose] Repo already exists at {repo_dir}, skipping clone.")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        print("[Bizarre Pose] Cloning ShuhongChen/bizarre-pose-estimator...")
        _run(["git", "clone", "--depth", "1",
              "https://github.com/ShuhongChen/bizarre-pose-estimator.git", str(repo_dir)])

    print()
    print("[Bizarre Pose] DATASET DOWNLOAD REQUIRED:")
    print("  1. Follow instructions in the repo README to download bizarre_pose_dataset.zip")
    print("  2. Extract into:")
    print(f"     {dest}/raw/")
    print()
    print("  Expected structure after extraction:")
    print(f"    {dest}/raw/bizarre_pose_dataset/")
    print()
    print("  Label content (COCO17 compatible):")
    print("    - 17 COCO keypoints: [x,y,v] * 17")
    print("    - bbox: [x, y, width, height] (auto-generated from segmentation)")
    print("    - Split: train 3200 / val 313 / test 487")
    print()
    print("  LIMITATIONS:")
    print("    - Single full-body character only (no multi-person, no partial-body)")
    print("    - Bbox auto-generated from masks (not manual)")


# ---------------------------------------------------------------------------
# Amateur Drawings
# ---------------------------------------------------------------------------

def download_amateur_drawings(data_root: Path) -> None:
    """Clone AnimatedDrawings repo and provide dataset download instructions.

    The full dataset (~50GB images + 275MB annotations) must be downloaded separately.
    """
    dest = data_root / "amateur_drawings"
    repo_dir = dest / "repo"

    if repo_dir.exists():
        print(f"[Amateur Drawings] Repo already exists at {repo_dir}, skipping clone.")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        print("[Amateur Drawings] Cloning facebookresearch/AnimatedDrawings...")
        _run(["git", "clone", "--depth", "1",
              "https://github.com/facebookresearch/AnimatedDrawings.git", str(repo_dir)])

    raw_dir = dest / "raw"
    raw_dir.mkdir(exist_ok=True)

    print()
    print("[Amateur Drawings] DATASET DOWNLOAD REQUIRED:")
    print("  1. Download annotations:")
    print("     amateur_drawings_annotations.json (~275MB)")
    print("  2. Download images:")
    print("     amateur_drawings.tar (~50GB)")
    print("  3. Place both in:")
    print(f"     {raw_dir}/")
    print("  4. Extract: tar xf amateur_drawings.tar")
    print()
    print("  Label content (NOT COCO compatible — conversion required):")
    print("    - 16 joints in custom hierarchical YAML format")
    print("    - Joint names: root, hip, torso, neck, {left,right}_{shoulder,elbow,hand,hip,knee,foot}")
    print("    - bbox: present in JSON")
    print("    - Segmentation masks: binary PNG per character")
    print()
    print("  GAPS vs COCO17:")
    print("    - NO facial keypoints (nose, eyes, ears) — will be set to [0,0,0] (unlabeled)")
    print("    - hand → wrist, foot → ankle (approximate mapping)")
    print("    - NO visibility flags — all mapped joints set to visibility=2")
    print("    - Annotations are model-predicted then user-accepted (noisy)")
    print()
    print("  Conversion will be done by:")
    print("    python -m pose_estimation.data.converters.amateur_drawings")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DATASETS = {
    "humanart": download_humanart,
    "bizarre_pose": download_bizarre_pose,
    "amateur_drawings": download_amateur_drawings,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public pose datasets")
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root directory for datasets (default: ./data)",
    )
    parser.add_argument(
        "--dataset", choices=[*DATASETS, "all"], default="all",
        help="Which dataset to download (default: all)",
    )
    args = parser.parse_args()

    args.data_root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "all":
        for name, fn in DATASETS.items():
            print(f"\n{'='*60}")
            print(f"  {name.upper()}")
            print(f"{'='*60}\n")
            fn(args.data_root)
    else:
        DATASETS[args.dataset](args.data_root)


if __name__ == "__main__":
    main()
