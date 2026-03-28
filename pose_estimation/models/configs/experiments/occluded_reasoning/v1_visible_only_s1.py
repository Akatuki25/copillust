"""V1-S1: Visible-only training (occluded keypoints removed from GT).

Tests: does excluding occluded keypoints from loss improve visible
keypoint accuracy? WACV 2024 reports visible-only gives better
PCK-visible on small datasets.

Method: v=1 keypoints are set to v=0 in the annotation file.
No architecture or codec change.
"""
_base_ = ["../curriculum/humanart_curriculum_s1.py"]

data_root = "data/merged_curriculum_s1/"
train_ann_file = "annotations/train_visible_only.json"

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann_file,
    ),
)

val_evaluator = dict(type="CocoMetric", ann_file=data_root + "annotations/val.json")
test_evaluator = val_evaluator
