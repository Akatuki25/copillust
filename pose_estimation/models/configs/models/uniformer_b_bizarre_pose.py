"""UniFormer-B + HeatmapHead fine-tune on Bizarre Pose.

Architecture: UniFormer-B backbone (Conv+Transformer hybrid) + HeatmapHead
Pretrained: COCO full model checkpoint
"""

_base_ = ["../base/base_bizarre_pose.py"]

custom_imports = dict(imports="projects.uniformer.models")
find_unused_parameters = True

codec = dict(
    type="MSRAHeatmap",
    input_size=(192, 256),
    heatmap_size=(48, 64),
    sigma=2,
)

load_from = ("https://download.openmmlab.com/mmpose/v1/projects/"
             "uniformer/top_down_256x192_global_base"
             "-1713bcd4_20230724.pth")

model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type="UniFormer",
        embed_dims=[64, 128, 320, 512],
        depths=[5, 8, 20, 7],
        head_dim=64,
        drop_path_rate=0.4,
        use_checkpoint=False,
        use_window=False,
        use_hybrid=False,
    ),
    head=dict(
        type="HeatmapHead",
        in_channels=512,
        out_channels=17,
        final_layer=dict(kernel_size=1),
        loss=dict(type="KeypointMSELoss", use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(flip_test=True, flip_mode="heatmap", shift_heatmap=True),
)

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
