"""T6: Wider sigma (6.0, 6.93) on curriculum S2."""
_base_ = ["../../experiments/curriculum/humanart_curriculum_s2.py"]

codec = dict(
    type="SimCCLabel",
    input_size=(192, 256),
    sigma=(6.0, 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

model = dict(
    head=dict(decoder=codec),
)

train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomBBoxTransform", shift_factor=0.2, scale_factor=(0.6, 1.4), rotate_factor=40),
    dict(type="TopdownAffine", input_size=(192, 256)),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
