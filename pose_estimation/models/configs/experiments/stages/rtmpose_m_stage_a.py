"""RTMPose-m Stage A config: Smoke test with Bizarre Pose dataset.

Stage A objectives:
- Verify loss descent on illustration domain
- Establish baseline improvement over pretrained COCO model

Base: RTMPose-m pretrained on body8, 256x192 input, SimCC head.
Data: Bizarre Pose (3200 train, 313 val).
Epochs: 10 (quick validation).
"""

# ==========================================================================
# Runtime (inlined from mmpose _base_/default_runtime.py)
# ==========================================================================
default_scope = "mmpose"

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

custom_hooks = [dict(type="SyncBuffersHook")]

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
# Model — load FULL pretrained RTMPose-m (backbone + head)
# ==========================================================================
load_from = ("https://download.openmmlab.com/mmpose/v1/projects/"
             "rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192"
             "-e48f03d0_20230504.pth")

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
        decoder=dict(
            type="SimCCLabel",
            input_size=(192, 256),
            sigma=(4.9, 5.66),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False,
        ),
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
# Pipelines
# ==========================================================================
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

val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=(192, 256)),
    dict(type="PackPoseInputs"),
]

# ==========================================================================
# Dataloaders
# ==========================================================================
train_dataloader = dict(
    batch_size=32,
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
    batch_size=32,
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
train_cfg = dict(max_epochs=10, val_interval=2, by_epoch=True)

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=5e-4, weight_decay=0.05),
)

param_scheduler = [
    dict(type="LinearLR", begin=0, end=3, start_factor=1e-5, by_epoch=True),
    dict(type="CosineAnnealingLR", begin=3, end=10, eta_min=1e-6, by_epoch=True),
]

auto_scale_lr = dict(base_batch_size=512)
