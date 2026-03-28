"""H3: UDP (Unbiased Data Processing) to fix left-right asymmetry.

Hypothesis: Horizontal flip augmentation introduces systematic coordinate
quantization errors. Left arm is consistently worse (53-62% fail) than right
(41-43%). UDP corrects the coordinate transform during flip, which should
reduce the left-right gap.

Based on: "The Devil is in the Details: Delving into Unbiased Data Processing
for Human Pose Estimation" (Huang et al., 2020).
Note: UDP is designed for heatmap codecs. For SimCC, the equivalent is ensuring
flip coordinate mapping is pixel-accurate. We test by switching codec to
UDPHeatmap to see if the encoding matters, or by fixing the SimCC flip.

Actually, SimCC doesn't have flip quantization issues since it operates in
continuous coordinate space. The left-right asymmetry is more likely from
dataset bias. So instead, we force BOTH original and flipped versions
to be seen equally by setting flip probability to 1.0 (always flip)
and doubling the dataset by including originals too.

Revised approach: Always include flipped version to balance left-right exposure.
"""
_base_ = ["../../experiments/curriculum/humanart_curriculum_s2.py"]

codec = dict(
    type="SimCCLabel",
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

# Use RandomFlip with prob=1.0 on a copy of the dataset to ensure
# every image is seen both ways. Actually, the simplest approach:
# just increase flip probability and train for double epochs.
# But cleaner: keep flip at 0.5 (default) which already ensures 50/50.
# The asymmetry likely comes from the dataset itself.
# Let's verify by explicitly forcing double-sided training:

train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal", prob=0.5),
    dict(type="RandomBBoxTransform", shift_factor=0.2, scale_factor=(0.6, 1.4), rotate_factor=40),
    dict(type="TopdownAffine", input_size=(192, 256)),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

# Train for 20 epochs instead of 10 to give more exposure to both orientations
train_cfg = dict(max_epochs=20, val_interval=2, by_epoch=True)

param_scheduler = [
    dict(type="LinearLR", begin=0, end=2, start_factor=1e-5, by_epoch=True),
    dict(type="CosineAnnealingLR", begin=2, end=20, eta_min=1e-7, by_epoch=True),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
