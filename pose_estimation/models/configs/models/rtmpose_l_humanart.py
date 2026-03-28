"""RTMPose-l (Human-Art pretrained) fine-tune on Bizarre Pose.

Architecture: CSPNeXt-l backbone + RTMCCHead (SimCC)
Pretrained: Human-Art dataset
"""

_base_ = ["../base/base_bizarre_pose.py"]

codec = dict(
    type="SimCCLabel",
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

load_from = ("https://download.openmmlab.com/mmpose/v1/projects/"
             "rtmposev1/rtmpose-l_8xb256-420e_humanart-256x192"
             "-389f2cb0_20230611.pth")

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
        deepen_factor=1.0,
        widen_factor=1.0,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="SiLU"),
    ),
    head=dict(
        type="RTMCCHead",
        in_channels=1024,
        out_channels=17,
        input_size=(192, 256),
        in_featuremap_size=(6, 8),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn="SiLU",
            use_rel_bias=False,
            pos_enc=False,
        ),
        loss=dict(
            type="KLDiscretLoss",
            use_target_weight=True,
            beta=10.0,
            label_softmax=True,
        ),
        decoder=codec,
    ),
    test_cfg=dict(flip_test=True),
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
