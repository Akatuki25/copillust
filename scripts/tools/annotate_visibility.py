"""Visibility annotation tool for Bizarre Pose dataset.

Reviews existing keypoint positions and lets you set visibility:
  v=2 (visible), v=1 (occluded but position known), v=0 (not in image)

Usage:
    python scripts/annotate_visibility.py
    python scripts/annotate_visibility.py --start 500   # resume from image 500

Controls:
    Click keypoint   : cycle visibility (2 → 1 → 0 → 2)
    Right click      : set to v=0 directly
    'a'              : accept current image (all visible by default), next
    'n'              : next image (save current)
    'p'              : previous image
    'q'              : save and quit
    '0'              : set ALL to v=0
    '1'              : set ALL to v=1
    '2'              : set ALL to v=2
    ESC              : quit without saving current

Progress is saved automatically. Resumes from last position.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

COCO17_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]

SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Colors for visibility states
VIS_COLORS = {
    2: (0, 255, 0),    # green = visible
    1: (0, 200, 255),  # yellow/orange = occluded
    0: (0, 0, 200),    # red = not visible
}

VIS_LABELS = {2: "visible", 1: "occluded", 0: "not_vis"}


class VisibilityAnnotator:
    def __init__(self, coco_json_path: Path, image_root: Path, output_path: Path):
        with open(coco_json_path) as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.output_path = output_path
        self.img_lookup = {img["id"]: img for img in self.data["images"]}
        self.ann_lookup = {ann["image_id"]: ann for ann in self.data["annotations"]}

        # Load progress if exists
        self.progress_file = output_path.parent / "visibility_progress.json"
        self.reviewed = set()
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                progress = json.load(f)
                self.reviewed = set(progress.get("reviewed", []))

        # If output already exists, load it as current state
        if output_path.exists():
            with open(output_path) as f:
                self.data = json.load(f)
                self.ann_lookup = {ann["image_id"]: ann for ann in self.data["annotations"]}

        self.sorted_ids = sorted(self.img_lookup.keys())
        self.current_idx = 0
        self.selected_kp = -1  # no selection
        self.move_mode = False  # position edit mode

    def _find_nearest_kp(self, x, y, kps, threshold=20):
        """Find nearest keypoint to click position."""
        min_dist = threshold
        best = -1
        for i in range(17):
            kx, ky = kps[i * 3], kps[i * 3 + 1]
            dist = ((x - kx) ** 2 + (y - ky) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                best = i
        return best

    def _draw(self, image, ann, img_info):
        """Draw image with keypoints colored by visibility."""
        vis = image.copy()
        h, w = vis.shape[:2]
        kps = ann["keypoints"]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw skeleton lines (only for visible pairs)
        for i, j in SKELETON_PAIRS:
            vi, vj = int(kps[i * 3 + 2]), int(kps[j * 3 + 2])
            if vi > 0 and vj > 0:
                pt1 = (int(kps[i * 3]), int(kps[i * 3 + 1]))
                pt2 = (int(kps[j * 3]), int(kps[j * 3 + 1]))
                color = (80, 80, 80) if vi == 1 or vj == 1 else (120, 120, 120)
                cv2.line(vis, pt1, pt2, color, 1)

        # Draw keypoints
        for i in range(17):
            x, y, v = kps[i * 3], kps[i * 3 + 1], int(kps[i * 3 + 2])
            color = VIS_COLORS[v]
            r = max(4, w // 120)

            if i == self.selected_kp:
                cv2.circle(vis, (int(x), int(y)), r + 4, (255, 255, 255), 2)

            cv2.circle(vis, (int(x), int(y)), r, color, -1)
            # Label
            label = f"{i}:{COCO17_NAMES[i][:4]}"
            cv2.putText(vis, label, (int(x) + r + 2, int(y) + 3), font, 0.3, color, 1)

        # Side panel
        panel_w = 200
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        # Stats
        v_counts = {0: 0, 1: 0, 2: 0}
        for i in range(17):
            v_counts[int(kps[i * 3 + 2])] += 1

        y_pos = 25
        cv2.putText(panel, f"Image {self.current_idx + 1}/{len(self.sorted_ids)}", (5, y_pos), font, 0.45, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(panel, f"Reviewed: {len(self.reviewed)}", (5, y_pos), font, 0.4, (180, 180, 180), 1)
        y_pos += 25

        for v, label in VIS_LABELS.items():
            color = VIS_COLORS[v]
            cv2.putText(panel, f"v={v} {label}: {v_counts[v]}", (5, y_pos), font, 0.4, color, 1)
            y_pos += 18

        y_pos += 15
        cv2.putText(panel, "--- Keypoints ---", (5, y_pos), font, 0.35, (150, 150, 150), 1)
        y_pos += 15

        for i in range(17):
            v = int(kps[i * 3 + 2])
            color = VIS_COLORS[v]
            marker = ["x", "?", "o"][v]
            text = f"{marker} {i:2d}:{COCO17_NAMES[i]}"
            if i == self.selected_kp:
                cv2.putText(panel, f"> {text}", (3, y_pos), font, 0.35, (255, 255, 255), 1)
            else:
                cv2.putText(panel, f"  {text}", (3, y_pos), font, 0.35, color, 1)
            y_pos += 16

        y_pos += 15
        cv2.putText(panel, "Click: cycle vis", (5, y_pos), font, 0.3, (140, 140, 140), 1)
        y_pos += 14
        cv2.putText(panel, "RClick: set v=0", (5, y_pos), font, 0.3, (140, 140, 140), 1)
        y_pos += 14
        cv2.putText(panel, "m: move selected kp", (5, y_pos), font, 0.3, (140, 140, 140), 1)
        y_pos += 14
        cv2.putText(panel, "a: accept  n: next", (5, y_pos), font, 0.3, (140, 140, 140), 1)
        y_pos += 14
        cv2.putText(panel, "p: prev  q: save+quit", (5, y_pos), font, 0.3, (140, 140, 140), 1)

        # Move mode indicator
        if self.move_mode and self.selected_kp >= 0:
            cv2.putText(panel, f"MOVE: {COCO17_NAMES[self.selected_kp]}", (5, h - 30),
                        font, 0.45, (0, 255, 255), 1)
            cv2.putText(panel, "Click to place", (5, h - 12), font, 0.35, (0, 255, 255), 1)

        # Reviewed indicator
        img_id = self.sorted_ids[self.current_idx]
        if img_id in self.reviewed:
            cv2.putText(panel, "[REVIEWED]", (5, h - 10), font, 0.45, (0, 200, 0), 1)

        return np.hstack([vis, panel])

    def _mouse_callback(self, event, mx, my, flags, param):
        img_id = self.sorted_ids[self.current_idx]
        ann = self.ann_lookup[img_id]
        kps = ann["keypoints"]

        # Adjust for display scale
        mx = int(mx / self._scale)
        my = int(my / self._scale)

        if mx >= self._img_w:
            return  # Click on panel

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.move_mode and self.selected_kp >= 0:
                # Move selected keypoint to click position
                kps[self.selected_kp * 3] = mx
                kps[self.selected_kp * 3 + 1] = my
                self.move_mode = False
            else:
                # Normal mode: find nearest and cycle visibility
                nearest = self._find_nearest_kp(mx, my, kps, threshold=max(20, self._img_w // 30))
                if nearest >= 0:
                    current_v = int(kps[nearest * 3 + 2])
                    new_v = {2: 1, 1: 0, 0: 2}[current_v]
                    kps[nearest * 3 + 2] = new_v
                    self.selected_kp = nearest
        elif event == cv2.EVENT_RBUTTONDOWN:
            nearest = self._find_nearest_kp(mx, my, kps, threshold=max(20, self._img_w // 30))
            if nearest >= 0:
                kps[nearest * 3 + 2] = 0
                self.selected_kp = nearest

    def save(self):
        """Save current state."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.data, f)
        with open(self.progress_file, "w") as f:
            json.dump({"reviewed": sorted(self.reviewed), "last_idx": self.current_idx}, f)
        print(f"Saved: {len(self.reviewed)}/{len(self.sorted_ids)} reviewed → {self.output_path}")

    def run(self, start_idx=0):
        """Main annotation loop."""
        self.current_idx = start_idx

        # Find first unreviewed if starting from 0
        if start_idx == 0 and self.reviewed:
            for i, img_id in enumerate(self.sorted_ids):
                if img_id not in self.reviewed:
                    self.current_idx = i
                    break

        window = "Visibility Annotator"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, self._mouse_callback)

        while True:
            img_id = self.sorted_ids[self.current_idx]
            img_info = self.img_lookup[img_id]
            ann = self.ann_lookup[img_id]

            img_path = self.image_root / img_info["file_name"]
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Cannot load: {img_path}")
                self.current_idx = min(self.current_idx + 1, len(self.sorted_ids) - 1)
                continue

            self._img_w = image.shape[1]
            self._img_h = image.shape[0]

            display = self._draw(image, ann, img_info)

            # Scale for display
            max_h = 800
            self._scale = min(max_h / display.shape[0], 1.0)
            if self._scale < 1.0:
                display = cv2.resize(display, None, fx=self._scale, fy=self._scale)

            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                self.save()
                break
            elif key == 27:  # ESC
                break
            elif key == ord("a") or key == ord("n"):
                # Accept/next: mark reviewed and save
                self.reviewed.add(img_id)
                self.current_idx = min(self.current_idx + 1, len(self.sorted_ids) - 1)
                self.selected_kp = -1
                self.move_mode = False
                self.save()
            elif key == ord("p"):
                self.current_idx = max(self.current_idx - 1, 0)
                self.selected_kp = -1
            elif key == ord("0"):
                kps = ann["keypoints"]
                for i in range(17):
                    kps[i * 3 + 2] = 0
            elif key == ord("1"):
                kps = ann["keypoints"]
                for i in range(17):
                    kps[i * 3 + 2] = 1
            elif key == ord("2"):
                kps = ann["keypoints"]
                for i in range(17):
                    kps[i * 3 + 2] = 2
            elif key == ord("m"):
                if self.selected_kp >= 0:
                    self.move_mode = True
            else:
                # Redraw on any other key or mouse event
                display = self._draw(image, ann, img_info)
                if self._scale < 1.0:
                    display = cv2.resize(display, None, fx=self._scale, fy=self._scale)
                cv2.imshow(window, display)

        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Annotate visibility for Bizarre Pose")
    parser.add_argument("--start", type=int, default=0, help="Start from image index")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    coco_json = project_root / f"data/bizarre_pose/coco/{args.split}.json"
    image_root = project_root / "data/bizarre_pose/raw/bizarre_pose_dataset/raw"
    output_path = project_root / f"data/bizarre_pose/coco/{args.split}_visibility.json"

    print(f"Input: {coco_json}")
    print(f"Output: {output_path}")
    print(f"Images: {image_root}")

    annotator = VisibilityAnnotator(coco_json, image_root, output_path)
    annotator.run(start_idx=args.start)


if __name__ == "__main__":
    main()
