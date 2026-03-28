"""H1: Per-keypoint loss weighting — 2x weight on elbows/wrists.

Hypothesis: Arms dominate failures (53-62%) while face is perfect (0%).
Equal loss weighting lets face gradients drown out arm learning signal.
Upweighting arm keypoints forces the model to prioritize arm accuracy.

Based on: RTMPose paper's per-keypoint weighting showing 2-5 AP on hard keypoints.
"""
_base_ = ["../../experiments/curriculum/humanart_curriculum_s2.py"]

# This requires modifying the dataset to provide keypoint_weights.
# MMPose's CocoDataset already supports this via the codec's target generation.
# However, SimCCLabel uses visibility as weight directly.
# The cleanest way is to multiply the visibility-based weight by a per-keypoint scale.
# We do this by creating a custom pipeline transform.

# For now, we override the model to use a modified loss that handles per-keypoint weighting.
# Actually, the simplest approach: modify the dataset's keypoint_weights in the pipeline.

# MMPose approach: use GenerateTarget with additional keypoint_weights override.
# The target_weight in the loss is [B, K, 1] from visibility.
# We can scale it per-keypoint by setting 'keypoint_weights' in the dataset metainfo.

# Simplest: use a custom codec wrapper. But actually, mmpose supports
# keypoint_weights at the dataset level via metainfo.

# Let's try setting it in the dataset:
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
    dict(type="GenerateTarget", encoder=codec, use_dataset_keypoint_weights=True),
    dict(type="PackPoseInputs"),
]

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline,
        # Per-keypoint weights: 2x on elbows(7,8), wrists(9,10), ears(3,4)
        metainfo=dict(
            keypoint_weights=[
                1.0,  # nose
                1.0,  # l_eye
                1.0,  # r_eye
                1.5,  # l_ear
                1.5,  # r_ear
                1.0,  # l_shoulder
                1.0,  # r_shoulder
                2.0,  # l_elbow
                2.0,  # r_elbow
                2.0,  # l_wrist
                2.0,  # r_wrist
                1.0,  # l_hip
                1.0,  # r_hip
                1.0,  # l_knee
                1.0,  # r_knee
                1.0,  # l_ankle
                1.0,  # r_ankle
            ],
        ),
    ),
)
