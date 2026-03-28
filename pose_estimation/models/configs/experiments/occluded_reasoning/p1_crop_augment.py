"""P1: RandomEdgesBlackout augmentation only (no architecture change).

Tests whether ProbPose's crop augmentation alone improves robustness
to partial visibility. ProbPose reports this as their dominant
improvement factor (+9.0 mAP on CropCOCO).

Architecture: standard RTMCCHead (no change from Curriculum S2).
Only the training pipeline is modified.

Comparison: Curriculum S2 baseline (same model, no blackout).
"""
_base_ = ["../curriculum/humanart_curriculum_s2.py"]

custom_imports = dict(
    imports=["pose_estimation.transforms.random_edges_blackout"],
    allow_failed_imports=False,
)

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
    # P1: RandomEdgesBlackout AFTER TopdownAffine, BEFORE GenerateTarget
    dict(type="RandomEdgesBlackout", prob=0.5, min_ratio=0.05, max_ratio=0.25, max_edges=2),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
