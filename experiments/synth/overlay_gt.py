"""Draw GT keypoints/skeleton from gt.json onto a render for visual verification."""
import json
import sys

import cv2

EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
VCOL = {0: (0, 0, 255), 1: (0, 165, 255), 2: (0, 200, 0)}  # v=0 red, v=1 orange, v=2 green

scene_dir, img_name = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "toon.png")
gt = json.load(open(f"{scene_dir}/gt.json"))
img = cv2.imread(f"{scene_dir}/{img_name}")
kps = gt["keypoints"]
for a, b in EDGES:
    if kps[a][2] > 0 and kps[b][2] > 0:
        cv2.line(img, (int(kps[a][0]), int(kps[a][1])), (int(kps[b][0]), int(kps[b][1])),
                 (255, 150, 0), 2)
for i, (x, y, v) in enumerate(kps):
    x, y = int(x), int(y)
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        cv2.circle(img, (x, y), 5, VCOL[v], -1)
        cv2.putText(img, str(i), (x + 6, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 200), 1)
out = f"{scene_dir}/gt_overlay.png"
cv2.imwrite(out, img)
print("wrote", out, "| v counts:", {v: sum(1 for k in kps if k[2] == v) for v in (0, 1, 2)})
