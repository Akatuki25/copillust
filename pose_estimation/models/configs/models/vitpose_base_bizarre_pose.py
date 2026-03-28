"""ViTPose-B (simple) fine-tune on Bizarre Pose.

Architecture: ViT-Base backbone + FeatureMapProcessor neck + HeatmapHead (UDPHeatmap codec)
Pretrained: COCO full model checkpoint
"""

_base_ = ["../base/base_bizarre_pose.py"]

# ViTPose uses LayerDecayOptimWrapper
custom_imports = dict(
    imports=["mmpose.engine.optim_wrappers.layer_decay_optim_wrapper"],
    allow_failed_imports=False,
)

codec = dict(
    type="UDPHeatmap",
    input_size=(192, 256),
    heatmap_size=(48, 64),
    sigma=2,
)

load_from = ("https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/"
             "topdown_heatmap/coco/td-hm_ViTPose-base-simple_8xb64-210e"
             "_coco-256x192-0b8234ea_20230407.pth")

model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type="mmpretrain.VisionTransformer",
        arch="base",
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.3,
        with_cls_token=False,
        out_type="featmap",
        patch_cfg=dict(padding=2),
    ),
    neck=dict(type="FeatureMapProcessor", scale_factor=4.0, apply_relu=True),
    head=dict(
        type="HeatmapHead",
        in_channels=768,
        out_channels=17,
        deconv_out_channels=[],
        deconv_kernel_sizes=[],
        final_layer=dict(kernel_size=3, padding=1),
        loss=dict(type="KeypointMSELoss", use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(flip_test=True, flip_mode="heatmap", shift_heatmap=False),
)

# Simple AdamW optimizer (same as other models for fair comparison)
optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=5e-4, weight_decay=0.05),
)

# Disable auto_scale_lr to prevent LR from being scaled down
auto_scale_lr = dict(base_batch_size=32)

# ViTPose needs use_udp=True in TopdownAffine
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
    dict(type="TopdownAffine", input_size=(192, 256), use_udp=True),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=(192, 256), use_udp=True),
    dict(type="PackPoseInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = val_dataloader
