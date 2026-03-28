"""Run pretrained RTMPose baseline inference on verification images.

This script runs the unmodified pretrained RTMPose-m on mydata/ images
to establish the baseline performance before any fine-tuning.

Results are saved as JSON and optional visualization images.

Usage:
    python -m pose_estimation.inference.baseline \\
        --image-dir ./mydata \\
        --output-dir ./experiments/baseline \\
        --config vendor/mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py \\
        --checkpoint path/to/rtmpose_m_coco.pth \\
        --visualize
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from pose_estimation.core.types import PoseResult
from pose_estimation.inference.predictor import Predictor
from pose_estimation.models.rtmpose_estimator import RTMPoseEstimator


def serialize_results(results: dict[str, list[PoseResult]]) -> dict:
    """Convert results to JSON-serializable format."""
    out: dict = {}
    for fname, poses in results.items():
        out[fname] = []
        for pose in poses:
            out[fname].append({
                "keypoints": [
                    {"x": kp.x, "y": kp.y, "confidence": kp.confidence, "visibility": kp.visibility}
                    for kp in pose.keypoints
                ],
                "bbox": list(pose.bbox.to_xywh()),
                "score": pose.score,
                "num_visible": pose.num_visible,
            })
    return out


def run_baseline(
    image_dir: Path,
    output_dir: Path,
    config: str,
    checkpoint: str | None,
    device: str | None = None,
    visualize: bool = False,
) -> dict[str, list[PoseResult]]:
    """Run baseline inference and save results.

    Args:
        image_dir: Directory with verification images (e.g., mydata/).
        output_dir: Where to save JSON results and visualizations.
        config: MMPose config path.
        checkpoint: Checkpoint path (None for default pretrained).
        device: Compute device.
        visualize: Whether to save visualization images.

    Returns:
        Dict mapping filename → PoseResult list.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    estimator = RTMPoseEstimator(config=config, checkpoint=checkpoint, device=device)
    predictor = Predictor(estimator)

    print(f"Running baseline inference on {image_dir}...")
    results = predictor.predict_directory(image_dir)

    # Save JSON results.
    json_path = output_dir / "baseline_results.json"
    serialized = serialize_results(results)
    with open(json_path, "w") as f:
        json.dump(serialized, f, indent=2)
    print(f"Results saved to {json_path}")

    # Print summary.
    print(f"\nBaseline summary ({len(results)} images):")
    for fname, poses in results.items():
        n_detected = len(poses)
        avg_visible = (
            sum(p.num_visible for p in poses) / n_detected if n_detected else 0
        )
        print(f"  {fname}: {n_detected} detection(s), avg visible kpts: {avg_visible:.1f}")

    # Optional visualization (requires evaluation/visualizer).
    if visualize:
        try:
            from pose_estimation.evaluation.visualizer import draw_pose_on_image

            vis_dir = output_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)

            for fname, poses in results.items():
                img_path = image_dir / fname
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                vis_image = draw_pose_on_image(image, poses)
                out_name = Path(fname).stem + "_baseline.jpg"
                cv2.imwrite(str(vis_dir / out_name), vis_image)

            print(f"Visualizations saved to {vis_dir}")
        except ImportError:
            print("Visualization skipped (evaluation.visualizer not available)")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RTMPose baseline on verification images")
    parser.add_argument("--image-dir", type=Path, default=Path("mydata"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/baseline"))
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    run_baseline(
        args.image_dir, args.output_dir, args.config, args.checkpoint,
        args.device, args.visualize,
    )


if __name__ == "__main__":
    main()
