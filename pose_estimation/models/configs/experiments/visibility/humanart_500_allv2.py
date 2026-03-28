"""RTMPose-m (Human-Art pretrained) on 500 images with all v=2 (no visibility correction)."""

_base_ = ["../../base/base_bizarre_pose.py"]

codec = dict(
    type="SimCCLabel",
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

load_from = ("https://download.openmmlab.com/mmpose/v1/projects/"
             "rtmposev1/rtmpose-m_8xb256-420e_humanart-256x192"
             "-8430627b_20230611.pth")

data_root = "data/merged_500_allv2/"

model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        _scope_="mmdet",
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="SiLU"),
    ),
    head=dict(
        type="RTMCCHead",
        in_channels=768,
        out_channels=17,
        input_size=(192, 256),
        in_featuremap_size=(6, 8),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256, s=128, expansion_factor=2,
            dropout_rate=0.0, drop_path=0.0,
            act_fn="SiLU", use_rel_bias=False, pos_enc=False,
        ),
        loss=dict(type="KLDiscretLoss", use_target_weight=True, beta=10.0, label_softmax=True),
        decoder=codec,
    ),
    test_cfg=dict(flip_test=True),
)

train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomBBoxTransform", shift_factor=0.2, scale_factor=(0.6, 1.4), rotate_factor=40),
    dict(type="TopdownAffine", input_size=(192, 256)),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_evaluator = dict(type="CocoMetric", ann_file=data_root + "annotations/val.json")
test_evaluator = val_evaluator
