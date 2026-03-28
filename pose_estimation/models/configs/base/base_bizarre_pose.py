"""Base config for Bizarre Pose fine-tuning experiments.

Shared settings:
- Runtime, logging, checkpoint hooks
- Dataset: Bizarre Pose (3200 train, 313 val)
- Augmentation: RandomFlip + RandomBBoxTransform
- Training: 10 epochs, AdamW lr=5e-4, cosine decay
- Batch size 32, 2 workers

Model configs should import this and override:
- model (architecture)
- load_from (pretrained checkpoint)
- codec (if different encoding is needed)
"""

# ==========================================================================
# Runtime
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
# Dataset
# ==========================================================================
dataset_type = "CocoDataset"
data_root = "data/merged/"
train_ann_file = "annotations/train.json"
val_ann_file = "annotations/val.json"
train_data_prefix = dict(img="images/")
val_data_prefix = dict(img="images/")

# ==========================================================================
# Default pipelines (256x192, model configs can override)
# ==========================================================================
_input_size = (192, 256)

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
    dict(type="TopdownAffine", input_size=_input_size),
    # GenerateTarget is added by model configs (codec-dependent)
]

val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=_input_size),
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
        pipeline=None,  # Set by model config
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
