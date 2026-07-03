"""H_wide_sigma_v1: Wider Gaussian targets for v=1 (occluded) keypoints.

Hypothesis: Occluded keypoint (v=1) annotations have inherent positional
uncertainty — the annotator guesses the position behind clothing. Training
with tight Gaussian targets on noisy GT positions overfits to annotation
noise. Using v1_sigma_scale=1.5 (sigma*1.5 for v=1) allows the model to
predict "approximately here" for occluded keypoints, reducing precision
penalty on uncertain annotations. Combined with v1_weight=0.5 (less loss
emphasis), this doubly reduces over-commitment to v=1 positions.
Expected: OKS@75 > 0.895.
"""

_base_ = ["../../base/base_bizarre_pose.py"]

custom_imports = dict(
    imports=[
        "pose_estimation.codecs.visibility_aware_simcc",
        "pose_estimation.codecs.wide_sigma_v1_simcc",
    ],
    allow_failed_imports=False,
)

codec = dict(
    type="WiderSigmaForOccludedSimCC",
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
    v1_weight=0.5,
    v1_sigma_scale=1.5,
)

load_from = "experiments/train/curriculum_s2/best_coco_AP_epoch_10.pth"

data_root = "data/merged_500_corrected/"

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

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.05),
)

param_scheduler = [
    dict(type="LinearLR", begin=0, end=2, start_factor=1e-5, by_epoch=True),
    dict(type="CosineAnnealingLR", begin=2, end=10, eta_min=1e-7, by_epoch=True),
]

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
