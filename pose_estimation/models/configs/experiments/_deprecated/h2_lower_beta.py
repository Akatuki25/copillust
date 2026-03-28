"""H2: Lower beta (5 instead of 10) in KL loss.

Hypothesis: beta=10 forces extremely sharp prediction distributions,
causing the model to be confidently wrong on uncertain keypoints (elbows/wrists).
Lower beta allows wider distributions when uncertain, producing better
gradients for position correction and more calibrated confidence scores.

Based on: SimCC paper's analysis of beta as temperature parameter;
"On Calibration of Modern Neural Networks" (Guo et al., 2017).
"""
_base_ = ["../../experiments/curriculum/humanart_curriculum_s2.py"]

model = dict(
    head=dict(
        loss=dict(beta=5.0),
    ),
)
