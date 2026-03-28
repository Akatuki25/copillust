"""Training wrapper for MMPose-based pose estimation.

Provides a clean interface over MMPose's tools/train.py with
stage management (A/B), device auto-detection, and experiment tracking.

Usage:
    python -m pose_estimation.training.trainer \\
        --config pose_estimation/models/configs/rtmpose_m_stage_a.py \\
        --work-dir experiments/rtmpose_m_stage_a \\
        --device auto

    python -m pose_estimation.training.trainer \\
        --config pose_estimation/models/configs/rtmpose_m_stage_b.py \\
        --work-dir experiments/rtmpose_m_stage_b \\
        --resume-from experiments/rtmpose_m_stage_a/best.pth
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def detect_device() -> str:
    """Auto-detect the best available training device."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def find_mmpose_train_script() -> Path:
    """Locate MMPose's tools/train.py."""
    candidates = [
        Path("vendor/mmpose/tools/train.py"),
        Path("../vendor/mmpose/tools/train.py"),
    ]
    # Also check if mmpose is installed and find its location.
    try:
        import mmpose

        pkg_dir = Path(mmpose.__file__).parent.parent
        candidates.append(pkg_dir / "tools" / "train.py")
    except ImportError:
        pass

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find mmpose/tools/train.py. "
        "Ensure MMPose is cloned to vendor/mmpose/"
    )


def run_training(
    config: str | Path,
    work_dir: str | Path,
    device: str = "auto",
    resume_from: str | Path | None = None,
    cfg_options: dict[str, str] | None = None,
) -> None:
    """Launch MMPose training.

    Args:
        config: Path to the MMPose config file.
        work_dir: Directory for checkpoints and logs.
        device: "auto", "cpu", "cuda", or "mps".
        resume_from: Optional checkpoint to resume from.
        cfg_options: Additional --cfg-options key=value pairs.
    """
    if device == "auto":
        device = detect_device()

    train_script = find_mmpose_train_script()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(train_script),
        str(config),
        "--work-dir", str(work_dir),
    ]

    if resume_from:
        cmd.extend(["--resume", str(resume_from)])

    # Build --cfg-options string.
    overrides: list[str] = []

    # Device handling for MMPose.
    if device == "cpu":
        overrides.append("device=cpu")
    elif device == "mps":
        overrides.append("device=mps")
    # cuda is default, no override needed.

    if cfg_options:
        for k, v in cfg_options.items():
            overrides.append(f"{k}={v}")

    if overrides:
        cmd.append("--cfg-options")
        cmd.extend(overrides)

    print(f"Training command:")
    print(f"  {' '.join(cmd)}")
    print(f"Device: {device}")
    print(f"Work dir: {work_dir}")
    print()

    env = os.environ.copy()
    if device == "mps":
        env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    # Add mmpose root to PYTHONPATH for projects/ imports (e.g. UniFormer)
    mmpose_root = str(Path(__file__).resolve().parent.parent.parent / "vendor" / "mmpose")
    env["PYTHONPATH"] = mmpose_root + ":" + env.get("PYTHONPATH", "")

    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pose estimation model")
    parser.add_argument("--config", type=str, required=True, help="MMPose config path")
    parser.add_argument("--work-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    run_training(
        config=args.config,
        work_dir=args.work_dir,
        device=args.device,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
