"""WiderSigmaForOccludedSimCC: wider Gaussian targets for v=1 keypoints.

Occluded keypoint annotations have inherent positional uncertainty.
Using wider Gaussian targets for v=1 keypoints reduces overfitting to
noisy GT positions, while v=2 keypoints retain tight targets.
"""

import numpy as np
from itertools import product
from typing import Optional, Tuple
from pose_estimation.codecs.visibility_aware_simcc import VisibilityAwareSimCC
from mmpose.registry import KEYPOINT_CODECS


@KEYPOINT_CODECS.register_module()
class WiderSigmaForOccludedSimCC(VisibilityAwareSimCC):
    """VisibilityAwareSimCC with per-visibility Gaussian sigma.

    For v=1 (occluded) keypoints, the Gaussian target is generated with
    sigma * v1_sigma_scale (wider distribution). For v=2 keypoints, the
    standard sigma is used.

    Args:
        v1_sigma_scale (float): Scale factor for sigma on v=1 keypoints.
            Default: 1.5 (50% wider than standard).
        **kwargs: Passed to VisibilityAwareSimCC.
    """

    def __init__(self, v1_sigma_scale: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self.v1_sigma_scale = v1_sigma_scale
        self.sigma_v1 = self.sigma * v1_sigma_scale

    def _generate_gaussian(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate Gaussian targets with per-visibility sigma."""
        N, K, _ = keypoints.shape
        w, h = self.input_size
        W = int(np.around(w * self.simcc_split_ratio))
        H = int(np.around(h * self.simcc_split_ratio))

        keypoints_split, keypoint_weights = self._map_coordinates(
            keypoints, keypoints_visible)

        target_x = np.zeros((N, K, W), dtype=np.float32)
        target_y = np.zeros((N, K, H), dtype=np.float32)

        # 3-sigma rule for bounds check (use base sigma for conservative bound)
        radius = self.sigma_v1 * 3  # larger radius covers both sigma cases

        x = np.arange(0, W, 1, dtype=np.float32)
        y = np.arange(0, H, 1, dtype=np.float32)

        for n, k in product(range(N), range(K)):
            vis_nk = float(keypoints_visible[n, k])
            if vis_nk < 0.5:
                continue

            mu = keypoints_split[n, k]

            left, top = mu - radius
            right, bottom = mu + radius + 1

            if left >= W or top >= H or right < 0 or bottom < 0:
                keypoint_weights[n, k] = 0
                continue

            mu_x, mu_y = mu

            # Use wider sigma for v=1 keypoints
            if 0.5 <= vis_nk < 1.5:
                sigma = self.sigma_v1
            else:
                sigma = self.sigma

            target_x[n, k] = np.exp(-((x - mu_x)**2) / (2 * sigma[0]**2))
            target_y[n, k] = np.exp(-((y - mu_y)**2) / (2 * sigma[1]**2))

        if self.normalize:
            norm_value = self.sigma * np.sqrt(np.pi * 2)
            target_x /= norm_value[0]
            target_y /= norm_value[1]

        return target_x, target_y, keypoint_weights
