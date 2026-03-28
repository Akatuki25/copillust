"""V1-S2: Mixed training with occluded at half weight.

Phase 2 of visibility-semantic curriculum:
- Load from V1-S1 checkpoint (visible-only baseline)
- Train on all 3200 images with corrected visibility
- v=2 (visible) weight = 1.0
- v=1 (occluded) weight = 0.5 (half the influence of visible)
- v=0 (absent) weight = 0.0

This treats occluded keypoints as "lower confidence teachers" rather
than equal-weight supervision. The model has already learned visible
keypoint positions in S1, so S2 adds occluded as supplementary signal.
"""
_base_ = ["../curriculum/humanart_curriculum_s2.py"]

custom_imports = dict(
    imports=["pose_estimation.transforms.visibility_weight_control"],
    allow_failed_imports=False,
)

load_from = "experiments/train/presence/v1_visible_only_s1/best_coco_AP_epoch_10.pth"

codec = dict(
    type="SimCCLabel",
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomBBoxTransform", shift_factor=0.2, scale_factor=(0.6, 1.4), rotate_factor=40),
    dict(type="TopdownAffine", input_size=(192, 256)),
    dict(type="GenerateTarget", encoder=codec),
    # v=1 (occluded) at half weight: supplementary signal, not equal teacher
    dict(type="VisibilityWeightControl", v0_weight=0.0, v1_weight=0.5, v2_weight=1.0),
    dict(type="PackPoseInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
