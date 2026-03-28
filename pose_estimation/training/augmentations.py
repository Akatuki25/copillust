"""Illustration-specific data augmentation definitions for MMPose.

These augmentation pipelines are designed to improve robustness on:
- Lineart / sketch images (grayscale, high contrast)
- Chibi / deformed characters (unusual proportions)
- Partial-body images (cropped frames)

Used in MMPose config files via the `train_pipeline` and `val_pipeline` fields.
"""

from __future__ import annotations

# Standard train pipeline for illustration pose estimation.
# Designed for RTMPose top-down with 256x192 input.
ILLUST_TRAIN_PIPELINE = [
    # Load image.
    dict(type="LoadImage"),
    # Get bbox from annotation.
    dict(type="GetBBoxCenterScale"),
    # Geometric augmentation.
    dict(type="RandomFlip", direction="horizontal"),
    dict(
        type="RandomBBoxTransform",
        shift_factor=0.2,      # More shift for partial-body tolerance
        scale_factor=(0.6, 1.4),  # Wider scale for chibi/deformed
        rotate_factor=40,      # Moderate rotation
    ),
    # Color augmentation — critical for lineart robustness.
    dict(
        type="ColorJitter",
        brightness=0.4,
        contrast=0.4,
        saturation=0.5,     # High saturation jitter: colored ↔ grayscale
        hue=0.1,
    ),
    # Random grayscale — forces model to not rely on color cues.
    # Essential for lineart images which have no color information.
    dict(
        type="RandomGrayscale",
        prob=0.3,            # 30% of training images become grayscale
    ),
    # Crop and resize to model input size.
    dict(type="TopdownAffine", input_size=(192, 256)),
    # Encode keypoints to heatmap / regression target.
    dict(
        type="GenerateTarget",
        encoder=dict(
            type="SimCCLabel",
            input_size=(192, 256),
            sigma=(4.9, 5.66),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False,
        ),
    ),
    # Format.
    dict(type="PackPoseInputs"),
]

# Validation pipeline — no augmentation.
ILLUST_VAL_PIPELINE = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=(192, 256)),
    dict(type="PackPoseInputs"),
]

# Aggressive augmentation variant for Stage B with more data.
ILLUST_TRAIN_PIPELINE_HEAVY = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(
        type="RandomBBoxTransform",
        shift_factor=0.25,
        scale_factor=(0.5, 1.5),
        rotate_factor=60,
    ),
    dict(
        type="ColorJitter",
        brightness=0.5,
        contrast=0.5,
        saturation=0.6,
        hue=0.15,
    ),
    dict(type="RandomGrayscale", prob=0.4),
    dict(type="TopdownAffine", input_size=(192, 256)),
    dict(
        type="GenerateTarget",
        encoder=dict(
            type="SimCCLabel",
            input_size=(192, 256),
            sigma=(4.9, 5.66),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False,
        ),
    ),
    dict(type="PackPoseInputs"),
]
