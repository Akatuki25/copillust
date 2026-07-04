"""R2b: continue fine-tuning Curriculum S2 on BP 3200 + synthetic sketches.

Tests whether 3D->sketch generative synthetic data (experiments/synth) improves
the sketch/chibi/truncation weaknesses that BP alone cannot teach.
Comparison baseline: r2b_bp_only.py (same schedule, BP data only) separates
the synth-data effect from mere extra epochs.

Data prep (see experiments/synth/README_windows.md):
  data/merged/annotations/train_r2b.json  (merge_r2b.py output)
  data/merged/images/synth -> experiments/synth/r2b/gen  (symlink)
"""

_base_ = ["../curriculum/humanart_curriculum_s2.py"]

load_from = "experiments/train/curriculum_s2/best_coco_AP_epoch_10.pth"

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    dataset=dict(ann_file="annotations/train_r2b.json"),
)

# lower LR: this is a 3rd-stage fine-tune on ~6x data
optim_wrapper = dict(optimizer=dict(type="AdamW", lr=5e-5, weight_decay=0.05))
param_scheduler = [
    dict(type="LinearLR", begin=0, end=1, start_factor=1e-2, by_epoch=True),
    dict(type="CosineAnnealingLR", begin=1, end=10, eta_min=1e-7, by_epoch=True),
]
