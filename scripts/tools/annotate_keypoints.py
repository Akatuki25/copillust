"""Simple keypoint annotation tool for mydata.

Usage:
    python scripts/annotate_keypoints.py mydata/lineart/xxx.jpeg
    python scripts/annotate_keypoints.py mydata/          # annotate all unannotated images

Controls:
    Left click  : place keypoint at cursor
    Right click : mark keypoint as not visible (skip)
    'u'         : undo last keypoint
    'r'         : restart current image
    'q'         : quit (saves progress)
    ESC         : quit without saving current image

Keypoints are saved to mydata/annotations.json in COCO format.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
    (200, 100, 50), (50, 100, 200), (100, 200, 50), (200, 50, 100),
    (50, 200, 100),
]


class KeypointAnnotator:
    def __init__(self, image_path: Path, annotations_file: Path):
        self.image_path = image_path
        self.annotations_file = annotations_file
        self.original = cv2.imread(str(image_path))
        if self.original is None:
            raise ValueError(f"Cannot load image: {image_path}")

        self.h, self.w = self.original.shape[:2]

        # Scale for display (max 900px height)
        self.display_max_h = 900
        self.display_max_w = 1400
        self.scale = min(self.display_max_h / self.h, self.display_max_w / self.w, 1.0)
        self.display_w = int(self.w * self.scale)
        self.display_h = int(self.h * self.scale)

        self.keypoints: list[tuple[float, float, int]] = []  # (x, y, visibility)
        self.current_idx = 0
        self.done = False
        self.cancelled = False

    def _to_display(self, x: float, y: float) -> tuple[int, int]:
        return int(x * self.scale), int(y * self.scale)

    def _to_original(self, dx: int, dy: int) -> tuple[float, float]:
        return dx / self.scale, dy / self.scale

    def _draw(self) -> np.ndarray:
        display = cv2.resize(self.original, (self.display_w, self.display_h))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw placed keypoints and skeleton
        for i, (x, y, v) in enumerate(self.keypoints):
            dx, dy = self._to_display(x, y)
            if v == 2:  # visible
                cv2.circle(display, (dx, dy), 5, COLORS[i % len(COLORS)], -1)
                cv2.circle(display, (dx, dy), 7, (255, 255, 255), 1)
                label = f"{i}:{COCO17_NAMES[i][:3]}"
                cv2.putText(display, label, (dx + 8, dy - 4), font, 0.35, (255, 255, 255), 1)
            elif v == 0:  # not visible
                cv2.circle(display, (dx, dy), 3, (100, 100, 100), -1)

        # Draw skeleton for placed keypoints
        for i, j in SKELETON_PAIRS:
            if i < len(self.keypoints) and j < len(self.keypoints):
                kp1 = self.keypoints[i]
                kp2 = self.keypoints[j]
                if kp1[2] == 2 and kp2[2] == 2:
                    pt1 = self._to_display(kp1[0], kp1[1])
                    pt2 = self._to_display(kp2[0], kp2[1])
                    cv2.line(display, pt1, pt2, COLORS[i % len(COLORS)], 2)

        # Side panel with reference skeleton + keypoint list
        panel_w = 260
        panel = np.zeros((self.display_h + 50, panel_w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)

        # Draw reference skeleton diagram on panel
        ref_cx, ref_cy = panel_w // 2, 130
        ref_scale = 1.8
        # Approximate COCO17 body layout positions for reference
        ref_positions = [
            (0, -50),    # 0: nose
            (-8, -56),   # 1: left_eye
            (8, -56),    # 2: right_eye
            (-18, -50),  # 3: left_ear
            (18, -50),   # 4: right_ear
            (-25, -25),  # 5: left_shoulder
            (25, -25),   # 6: right_shoulder
            (-40, 0),    # 7: left_elbow
            (40, 0),     # 8: right_elbow
            (-50, 25),   # 9: left_wrist
            (50, 25),    # 10: right_wrist
            (-15, 20),   # 11: left_hip
            (15, 20),    # 12: right_hip
            (-20, 50),   # 13: left_knee
            (20, 50),    # 14: right_knee
            (-22, 80),   # 15: left_ankle
            (22, 80),    # 16: right_ankle
        ]

        ref_pts = [(int(ref_cx + x * ref_scale), int(ref_cy + y * ref_scale)) for x, y in ref_positions]

        # Draw reference skeleton lines
        for i, j in SKELETON_PAIRS:
            cv2.line(panel, ref_pts[i], ref_pts[j], (80, 80, 80), 1)

        # Draw reference points
        for i, pt in enumerate(ref_pts):
            if i < self.current_idx:
                # Already placed
                cv2.circle(panel, pt, 5, (80, 80, 80), -1)
            elif i == self.current_idx:
                # Current target - highlight
                cv2.circle(panel, pt, 8, (0, 255, 255), 2)
                cv2.circle(panel, pt, 4, (0, 255, 255), -1)
            else:
                # Future
                cv2.circle(panel, pt, 3, (60, 60, 60), -1)

        # Draw keypoint list below skeleton
        list_y_start = ref_cy + 100 * int(ref_scale) + 10
        for i, name in enumerate(COCO17_NAMES):
            y_pos = list_y_start + i * 20
            if y_pos > self.display_h + 40:
                break

            if i < len(self.keypoints):
                v = self.keypoints[i][2]
                if v == 2:
                    color = (0, 200, 0)
                    marker = "o"
                else:
                    color = (100, 100, 100)
                    marker = "x"
                cv2.putText(panel, f"{marker} {i:2d}: {name}", (10, y_pos), font, 0.4, color, 1)
            elif i == self.current_idx:
                cv2.putText(panel, f"> {i:2d}: {name}", (10, y_pos), font, 0.4, (0, 255, 255), 1)
            else:
                cv2.putText(panel, f"  {i:2d}: {name}", (10, y_pos), font, 0.4, (120, 120, 120), 1)

        # Status bar at top
        bar_h = 50
        bar = np.zeros((bar_h, self.display_w + panel_w, 3), dtype=np.uint8)
        if self.current_idx < 17:
            name = COCO17_NAMES[self.current_idx]
            text = f"[{self.current_idx+1}/17] Click: {name}"
            cv2.putText(bar, text, (10, 22), font, 0.6, (0, 255, 255), 1)
            cv2.putText(bar, "Left=place  Right=skip(not visible)  u=undo  r=restart  q=save&quit", (10, 42), font, 0.4, (180, 180, 180), 1)
        else:
            cv2.putText(bar, "All 17 keypoints done! Press any key to save.", (10, 30), font, 0.6, (0, 255, 255), 1)

        # Combine: bar on top, image+panel below
        body = np.hstack([display, panel[:self.display_h, :, :]])
        return np.vstack([bar, body])

    def _mouse_callback(self, event, x, y, flags, param):
        if self.current_idx >= 17:
            return

        # Adjust y for status bar
        y -= 50
        if y < 0 or x >= self.display_w:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Place keypoint (visible)
            ox, oy = self._to_original(x, y)
            self.keypoints.append((ox, oy, 2))
            self.current_idx += 1
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Skip keypoint (not visible)
            self.keypoints.append((0, 0, 0))
            self.current_idx += 1

    def run(self) -> bool:
        """Run annotation. Returns True if completed, False if cancelled."""
        window_name = f"Annotate: {self.image_path.name}"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        while True:
            display = self._draw()
            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                if self.current_idx == 17:
                    self.done = True
                break
            elif key == 27:  # ESC
                self.cancelled = True
                break
            elif key == ord('u') and self.keypoints:
                self.keypoints.pop()
                self.current_idx = max(0, self.current_idx - 1)
            elif key == ord('r'):
                self.keypoints.clear()
                self.current_idx = 0
            elif self.current_idx >= 17:
                self.done = True
                break

        cv2.destroyWindow(window_name)
        return self.done

    def get_coco_keypoints(self) -> list[float]:
        """Return keypoints in COCO format: [x1,y1,v1, x2,y2,v2, ...]"""
        result = []
        for x, y, v in self.keypoints:
            result.extend([round(x, 1), round(y, 1), v])
        return result


def load_annotations(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"images": [], "annotations": []}


def save_annotations(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved to {path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = Path(sys.argv[1])
    mydata_root = Path("mydata")
    annotations_file = mydata_root / "annotations.json"

    # Collect images to annotate
    if target.is_file():
        image_paths = [target]
    elif target.is_dir():
        image_paths = sorted(
            list(target.rglob("*.jpeg")) +
            list(target.rglob("*.jpg")) +
            list(target.rglob("*.png"))
        )
    else:
        print(f"Not found: {target}")
        sys.exit(1)

    # Load existing annotations
    data = load_annotations(annotations_file)
    annotated_files = {img["file_name"] for img in data["images"]}

    # Filter out already annotated
    remaining = []
    for p in image_paths:
        rel = str(p.relative_to(mydata_root)) if p.is_relative_to(mydata_root) else p.name
        if rel not in annotated_files:
            remaining.append((p, rel))

    if not remaining:
        print("All images already annotated!")
        print(f"Total: {len(data['images'])} annotations in {annotations_file}")
        sys.exit(0)

    print(f"Images to annotate: {len(remaining)} (already done: {len(annotated_files)})")
    print("Controls: left-click=place, right-click=skip(not visible), u=undo, r=restart, q=save&quit, ESC=cancel")
    print()

    next_img_id = max((img["id"] for img in data["images"]), default=0) + 1
    next_ann_id = max((ann["id"] for ann in data["annotations"]), default=0) + 1

    for i, (img_path, rel_path) in enumerate(remaining):
        print(f"[{i+1}/{len(remaining)}] {rel_path}")
        annotator = KeypointAnnotator(img_path, annotations_file)
        completed = annotator.run()

        if annotator.cancelled:
            print("  Cancelled. Saving progress...")
            save_annotations(data, annotations_file)
            break

        if completed:
            kps = annotator.get_coco_keypoints()
            num_visible = sum(1 for j in range(2, len(kps), 3) if kps[j] == 2)

            img_entry = {
                "id": next_img_id,
                "file_name": rel_path,
                "width": annotator.w,
                "height": annotator.h,
            }
            ann_entry = {
                "id": next_ann_id,
                "image_id": next_img_id,
                "category_id": 1,
                "keypoints": kps,
                "num_keypoints": num_visible,
                "bbox": [0, 0, annotator.w, annotator.h],
                "area": annotator.w * annotator.h,
                "iscrowd": 0,
            }
            data["images"].append(img_entry)
            data["annotations"].append(ann_entry)
            next_img_id += 1
            next_ann_id += 1
            print(f"  Saved: {num_visible}/17 visible keypoints")

            # Auto-save after each image
            save_annotations(data, annotations_file)
        else:
            print("  Skipped (incomplete)")

    save_annotations(data, annotations_file)
    print(f"\nDone! Total annotations: {len(data['images'])}")


if __name__ == "__main__":
    main()
