"""T4: CoarseDropout augmentation on curriculum S2."""
_base_ = ["../../experiments/curriculum/humanart_curriculum_s2.py"]

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
    dict(type="RandomBBoxTransform", shift_factor=0.2, scale_factor=(0.6, 1.4), rotate_factor=40),
    dict(type="TopdownAffine", input_size=(192, 256)),
    dict(
        type="Albumentation",
        transforms=[
            dict(type="CoarseDropout", max_holes=1, max_height=0.3, max_width=0.3,
                 min_holes=1, min_height=0.1, min_width=0.1, p=0.5),
        ],
    ),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
