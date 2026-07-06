"""R2b ablation v2: BP 3200 + synth 3200 (cs0.9 only, gate PCK@0.2>=0.95, 1:1).

v1 (r2b_mixed.py, 82% synth incl. cs0.55) degraded mydata across all
categories (0.548 -> 0.453 OKS@50) — synthetic domain pull overwhelmed the
real-illustration statistics. v2 tests a clean, balanced mix.
"""

_base_ = ["../curriculum/humanart_curriculum_s2.py"]

load_from = "experiments/train/curriculum_s2/best_coco_AP_epoch_10.pth"

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    dataset=dict(ann_file="annotations/train_r2b_v2.json"),
)

optim_wrapper = dict(optimizer=dict(type="AdamW", lr=5e-5, weight_decay=0.05))
param_scheduler = [
    dict(type="LinearLR", begin=0, end=1, start_factor=1e-2, by_epoch=True),
    dict(type="CosineAnnealingLR", begin=1, end=10, eta_min=1e-7, by_epoch=True),
]
