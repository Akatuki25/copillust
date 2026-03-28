"""3L+mask curriculum S1: clean 891 images."""
_base_ = ["./occluded_3layer.py"]

data_root = "data/merged_curriculum_s1/"

load_from = ("https://download.openmmlab.com/mmpose/v1/projects/"
             "rtmposev1/rtmpose-m_8xb256-420e_humanart-256x192"
             "-8430627b_20230611.pth")

train_cfg = dict(max_epochs=10, val_interval=2, by_epoch=True)

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=5e-4, weight_decay=0.05),
)

param_scheduler = [
    dict(type="LinearLR", begin=0, end=3, start_factor=1e-5, by_epoch=True),
    dict(type="CosineAnnealingLR", begin=3, end=10, eta_min=1e-6, by_epoch=True),
]

val_evaluator = dict(type="CocoMetric", ann_file=data_root + "annotations/val.json")
test_evaluator = val_evaluator
