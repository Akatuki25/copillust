"""SimCCLabel with configurable visibility-to-weight mapping.

Subclasses mmpose's SimCCLabel to allow controlling how visibility
values map to loss weights, without modifying the vendor code.
"""

import numpy as np
from typing import Optional, Tuple
from mmpose.codecs import SimCCLabel
from mmpose.registry import KEYPOINT_CODECS


@KEYPOINT_CODECS.register_module()
class VisibilityAwareSimCC(SimCCLabel):
    """SimCCLabel with configurable visibility weight mapping.

    Args:
        v1_weight (float): Weight for occluded keypoints (v=1).
            Default: 1.0 (same as standard SimCCLabel).
            Set to 0.0 for visible-only training.
        **kwargs: All other args passed to SimCCLabel.
    """

    def __init__(self, v1_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.v1_weight = v1_weight

    def _map_coordinates(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Override to apply custom visibility-to-weight mapping."""
        keypoints_split, keypoint_weights = super()._map_coordinates(
            keypoints, keypoints_visible)

        # Remap v=1 weights
        # keypoint_weights comes from keypoints_visible.copy()
        # v=0 → 0, v=1 → 1, v=2 → 2 by default
        # We override v=1 → self.v1_weight
        if keypoints_visible is not None:
            vis = keypoints_visible.copy()
            if vis.ndim == 3:
                vis = vis.squeeze(-1)
            mask_v1 = (vis >= 0.5) & (vis < 1.5)
            keypoint_weights[mask_v1] = self.v1_weight

        return keypoints_split, keypoint_weights
