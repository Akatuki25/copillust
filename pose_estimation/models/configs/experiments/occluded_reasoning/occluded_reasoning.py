"""Occlusion-aware pose reasoning on curriculum S2.

Based on: "Rethinking Visibility in Human Pose Estimation:
Occluded Pose Reasoning via Transformers" (Sun et al., WACV 2024)

Uses OccludedRTMCCHead which adds:
1. VisNet: predicts per-keypoint visibility
2. Masking: attenuates occluded features before GAU
3. GAU reasons from visible keypoints to infer occluded ones
4. Auxiliary visibility BCE loss (weight=0.33)

Hypothesis: The model currently treats all keypoints equally in GAU.
By predicting which keypoints are occluded and masking their features,
the GAU is forced to infer occluded arm positions from visible
shoulder/torso/hand features. This directly addresses the arm failure
pattern (53-62% failure) caused by clothing hiding arm contours.

Requires: corrected visibility annotations (v=0/1/2) which we have
for all 3200 Bizarre Pose images.
"""

_base_ = ["../curriculum/humanart_curriculum_s2.py"]

custom_imports = dict(
    imports=["pose_estimation.models.heads.occluded_rtmcc_head"],
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

model = dict(
    head=dict(
        type="OccludedRTMCCHead",
        in_channels=768,
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
        vis_loss_weight=0.33,
        occ_attenuation=0.01,
        mix_gt_prob=0.5,
    ),
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
