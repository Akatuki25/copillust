"""RandomEdgesBlackout: randomly black out edges of the image.

Based on ProbPose (CVPR 2025) - creates training examples where keypoints
fall outside the visible area, providing negative examples for presence
prediction. Also serves as a strong crop augmentation that improves
robustness to partial visibility.

This transform runs AFTER TopdownAffine (on the cropped/resized image).
It blacks out a random strip from one or more edges, and sets
keypoints_visible=0 for any keypoint that falls in the blacked-out region.
"""

from typing import Dict, Optional
import numpy as np
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomEdgesBlackout(BaseTransform):
    """Randomly black out edges of the input image.

    Args:
        prob (float): Probability of applying the transform. Default: 0.5.
        min_ratio (float): Minimum ratio of edge to black out. Default: 0.05.
        max_ratio (float): Maximum ratio of edge to black out. Default: 0.25.
        max_edges (int): Maximum number of edges to black out. Default: 2.
    """

    def __init__(self,
                 prob: float = 0.5,
                 min_ratio: float = 0.05,
                 max_ratio: float = 0.25,
                 max_edges: int = 2):
        self.prob = prob
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.max_edges = max_edges

    def transform(self, results: Dict) -> Optional[Dict]:
        if np.random.rand() > self.prob:
            return results

        img = results['img']
        h, w = img.shape[:2]

        # Choose which edges to black out (0=top, 1=bottom, 2=left, 3=right)
        n_edges = np.random.randint(1, self.max_edges + 1)
        edges = np.random.choice(4, size=n_edges, replace=False)

        blackout_regions = []  # (x_min, y_min, x_max, y_max)

        for edge in edges:
            ratio = np.random.uniform(self.min_ratio, self.max_ratio)
            if edge == 0:  # top
                cut = int(h * ratio)
                img[:cut, :] = 0
                blackout_regions.append((0, 0, w, cut))
            elif edge == 1:  # bottom
                cut = int(h * ratio)
                img[h - cut:, :] = 0
                blackout_regions.append((0, h - cut, w, h))
            elif edge == 2:  # left
                cut = int(w * ratio)
                img[:, :cut] = 0
                blackout_regions.append((0, 0, cut, h))
            elif edge == 3:  # right
                cut = int(w * ratio)
                img[:, w - cut:] = 0
                blackout_regions.append((w - cut, 0, w, h))

        results['img'] = img

        # Update keypoint visibility: set to 0 for keypoints in blacked-out regions
        if 'keypoints' in results:
            keypoints = results['keypoints']  # (N, K, 2)
            keypoints_visible = results.get('keypoints_visible',
                                             np.ones(keypoints.shape[:2], dtype=np.float32))

            for n in range(keypoints.shape[0]):
                for k in range(keypoints.shape[1]):
                    x, y = keypoints[n, k]
                    for (rx_min, ry_min, rx_max, ry_max) in blackout_regions:
                        if rx_min <= x < rx_max and ry_min <= y < ry_max:
                            keypoints_visible[n, k] = 0
                            break

            results['keypoints_visible'] = keypoints_visible

        return results
