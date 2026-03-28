"""3-layer GAU without masking.

Isolate the effect of deeper inter-keypoint reasoning
from the masking mechanism.
"""
_base_ = ["./occluded_3layer.py"]

model = dict(
    head=dict(
        occ_attenuation=1.0,  # 1.0 = no attenuation, masking disabled
        vis_loss_weight=0.0,  # disable visibility loss
        mix_gt_prob=0.0,      # never use GT visibility
    ),
)
