"""HRNet-W48 + HeatmapHead fine-tune on Bizarre Pose.

Architecture: HRNet-W48 backbone + 2D HeatmapHead (MSRAHeatmap codec)
Pretrained: COCO full model checkpoint
"""

_base_ = ["../base/base_bizarre_pose.py"]

# Codec (heatmap-based, different from SimCC)
codec = dict(
    type="MSRAHeatmap",
    input_size=(192, 256),
    heatmap_size=(48, 64),
    sigma=2,
)

# Model
load_from = ("https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/"
             "topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192"
             "-0e67c616_20220913.pth")

model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type="HRNet",
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1, num_branches=1,
                block="BOTTLENECK", num_blocks=(4,), num_channels=(64,),
            ),
            stage2=dict(
                num_modules=1, num_branches=2,
                block="BASIC", num_blocks=(4, 4), num_channels=(48, 96),
            ),
            stage3=dict(
                num_modules=4, num_branches=3,
                block="BASIC", num_blocks=(4, 4, 4), num_channels=(48, 96, 192),
            ),
            stage4=dict(
                num_modules=3, num_branches=4,
                block="BASIC", num_blocks=(4, 4, 4, 4), num_channels=(48, 96, 192, 384),
            ),
        ),
    ),
    head=dict(
        type="HeatmapHead",
        in_channels=48,
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type="KeypointMSELoss", use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(flip_test=True, flip_mode="heatmap", shift_heatmap=True),
)

# Override train pipeline with heatmap codec
train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(
        type="RandomBBoxTransform",
        shift_factor=0.2,
        scale_factor=(0.6, 1.4),
        rotate_factor=40,
    ),
    dict(type="TopdownAffine", input_size=(192, 256)),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
