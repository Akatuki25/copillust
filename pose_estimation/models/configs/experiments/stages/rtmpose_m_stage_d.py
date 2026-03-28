"""RTMPose-m Stage D: Optimized training on Bizarre Pose only.

Key changes from Stage A:
1. Input resolution 384x288 (was 256x192) — 2.25x more pixels for fine joints
2. Official augmentation pipeline (RandomHalfBody, CoarseDropout, Blur, YOLOXHSVRandomAug)
3. EMA for training stability with small dataset
4. Two-stage pipeline (aggressive aug → reduced aug for final epochs)
5. Backbone lr_mult=0.1 to prevent overfitting
6. Fixed auto_scale_lr to match actual batch size
7. Pretrained from body7 384x288 checkpoint

Data: Bizarre Pose (3200 train, 313 val).
Epochs: 30.
"""

# ==========================================================================
# Runtime
# ==========================================================================
default_scope = "mmpose"

max_epochs = 30
stage2_num_epochs = 8
base_lr = 2e-4

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=10),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=2,
        save_best="coco/AP",
        rule="greater",
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="PoseVisualizationHook", enable=False),
)

custom_hooks = [
    dict(type="SyncBuffersHook"),
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type="mmdet.PipelineSwitchHook",
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=None,  # Will be set below after train_pipeline_stage2 is defined
    ),
]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="gloo"),
)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="PoseLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True, num_digits=6)
log_level = "INFO"
resume = False
backend_args = dict(backend="local")

val_cfg = dict()
test_cfg = dict()

# ==========================================================================
# Model — 384x288, pretrained from body7
# ==========================================================================
load_from = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-384x288-65e718c4_20230504.pth"

codec = dict(
    type="SimCCLabel",
    input_size=(288, 384),
    sigma=(6.0, 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

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
        input_size=(288, 384),
        in_featuremap_size=(9, 12),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.1,
            drop_path=0.1,
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

# ==========================================================================
# Dataset
# ==========================================================================
dataset_type = "CocoDataset"
data_root = "data/merged/"
train_ann_file = "annotations/train.json"
val_ann_file = "annotations/val.json"
train_data_prefix = dict(img="images/")
val_data_prefix = dict(img="images/")

# ==========================================================================
# Pipelines — Official RTMPose augmentation adapted for illustrations
# ==========================================================================
train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomHalfBody"),
    dict(
        type="RandomBBoxTransform",
        scale_factor=(0.5, 1.5),
        rotate_factor=90,
    ),
    dict(type="TopdownAffine", input_size=(288, 384)),
    dict(type="mmdet.YOLOXHSVRandomAug"),
    dict(type="PhotometricDistortion"),
    dict(
        type="Albumentation",
        transforms=[
            dict(type="Blur", p=0.1),
            dict(type="MedianBlur", p=0.1),
            dict(
                type="CoarseDropout",
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0,
            ),
        ],
    ),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

train_pipeline_stage2 = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomHalfBody"),
    dict(
        type="RandomBBoxTransform",
        shift_factor=0.0,
        scale_factor=(0.5, 1.5),
        rotate_factor=90,
    ),
    dict(type="TopdownAffine", input_size=(288, 384)),
    dict(type="mmdet.YOLOXHSVRandomAug"),
    dict(
        type="Albumentation",
        transforms=[
            dict(type="Blur", p=0.1),
            dict(type="MedianBlur", p=0.1),
            dict(
                type="CoarseDropout",
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5,
            ),
        ],
    ),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

# Fix PipelineSwitchHook reference
custom_hooks[2] = dict(
    type="mmdet.PipelineSwitchHook",
    switch_epoch=max_epochs - stage2_num_epochs,
    switch_pipeline=train_pipeline_stage2,
)

val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=(288, 384)),
    dict(type="PackPoseInputs"),
]

# ==========================================================================
# Dataloaders
# ==========================================================================
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=train_data_prefix,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=val_data_prefix,
        pipeline=val_pipeline,
        test_mode=True,
    ),
)

test_dataloader = val_dataloader

# ==========================================================================
# Evaluator
# ==========================================================================
val_evaluator = dict(type="CocoMetric", ann_file=data_root + val_ann_file)
test_evaluator = val_evaluator

# ==========================================================================
# Training schedule
# ==========================================================================
train_cfg = dict(max_epochs=max_epochs, val_interval=2, by_epoch=True)

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1),
        }
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-5, by_epoch=False, begin=0, end=500),
    dict(
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# Match actual batch size so configured LR is effective LR
auto_scale_lr = dict(base_batch_size=16)
