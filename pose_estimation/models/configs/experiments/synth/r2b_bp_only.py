"""R2b control: identical schedule to r2b_mixed.py but BP 3200 only.

Separates "synthetic data helped" from "more fine-tuning epochs helped".
"""

_base_ = ["../curriculum/humanart_curriculum_s2.py"]

load_from = "experiments/train/curriculum_s2/best_coco_AP_epoch_10.pth"

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    dataset=dict(ann_file="annotations/train.json"),
)

optim_wrapper = dict(optimizer=dict(type="AdamW", lr=5e-5, weight_decay=0.05))
param_scheduler = [
    dict(type="LinearLR", begin=0, end=1, start_factor=1e-2, by_epoch=True),
    dict(type="CosineAnnealingLR", begin=1, end=10, eta_min=1e-7, by_epoch=True),
]
