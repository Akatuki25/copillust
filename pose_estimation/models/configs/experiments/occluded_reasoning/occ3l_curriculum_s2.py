"""3L+mask curriculum S2: all 3200 images from S1 checkpoint."""
_base_ = ["./occluded_3layer.py"]

load_from = "experiments/train/techniques/occ3l_curriculum_s1/best_coco_AP_epoch_10.pth"

data_root = "data/merged_500_corrected/"  # full 3200 corrected

train_cfg = dict(max_epochs=10, val_interval=2, by_epoch=True)

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.05),
)

param_scheduler = [
    dict(type="LinearLR", begin=0, end=2, start_factor=1e-5, by_epoch=True),
    dict(type="CosineAnnealingLR", begin=2, end=10, eta_min=1e-7, by_epoch=True),
]

val_evaluator = dict(type="CocoMetric", ann_file=data_root + "annotations/val.json")
test_evaluator = val_evaluator
