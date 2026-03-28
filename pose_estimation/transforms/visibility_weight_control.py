"""VisibilityWeightControl: override target_weight based on visibility.

Controls how much occluded (v=1) keypoints contribute to the loss,
independent of the codec's default behavior (v=value → weight=value).

This runs AFTER GenerateTarget in the pipeline.
"""

import numpy as np
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class VisibilityWeightControl(BaseTransform):
    """Override keypoint_weights based on visibility values.

    By default, mmpose sets keypoint_weights = visibility value directly
    (v=0→0, v=1→1, v=2→2). This transform remaps those weights.

    Args:
        v0_weight (float): Weight for v=0 (not labeled). Default: 0.0.
        v1_weight (float): Weight for v=1 (occluded). Default: 1.0.
        v2_weight (float): Weight for v=2 (visible). Default: 2.0.
    """

    def __init__(self,
                 v0_weight: float = 0.0,
                 v1_weight: float = 1.0,
                 v2_weight: float = 2.0):
        self.v0_weight = v0_weight
        self.v1_weight = v1_weight
        self.v2_weight = v2_weight

    def transform(self, results: dict) -> dict:
        if 'keypoint_weights' not in results:
            if not hasattr(self, '_warned_no_weights'):
                print(f"[VisibilityWeightControl] keypoint_weights not in results. Keys: {list(results.keys())}")
                self._warned_no_weights = True
            return results

        visible = results.get('keypoints_visible', None)
        if visible is None:
            if not hasattr(self, '_warned_no_vis'):
                print(f"[VisibilityWeightControl] keypoints_visible not in results. Keys: {list(results.keys())}")
                self._warned_no_vis = True
            return results

        if not hasattr(self, '_debug_once'):
            print(f"[VisibilityWeightControl] keypoints_visible shape={visible.shape} unique={np.unique(visible)}")
            print(f"[VisibilityWeightControl] keypoint_weights type={type(results['keypoint_weights'])}")
            if isinstance(results['keypoint_weights'], np.ndarray):
                print(f"[VisibilityWeightControl] keypoint_weights shape={results['keypoint_weights'].shape} unique={np.unique(results['keypoint_weights'])}")
            self._debug_once = True

        # visible can be (N, K) or (N, K, 1)
        vis = visible.squeeze(-1) if visible.ndim == 3 else visible

        # Build new weight array from visibility values
        new_weights = np.zeros_like(vis, dtype=np.float32)
        new_weights[vis < 0.5] = self.v0_weight   # v=0
        new_weights[(vis >= 0.5) & (vis < 1.5)] = self.v1_weight  # v=1
        new_weights[vis >= 1.5] = self.v2_weight   # v=2

        # keypoint_weights may be a list (multi-encoder) or array
        existing = results['keypoint_weights']
        if isinstance(existing, list):
            results['keypoint_weights'] = [new_weights for _ in existing]
        else:
            results['keypoint_weights'] = new_weights

        return results
