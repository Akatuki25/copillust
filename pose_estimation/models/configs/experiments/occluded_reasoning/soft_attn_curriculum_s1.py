"""3L GAU + soft attention masking, curriculum S1 (clean 891 images).

Hypothesis: Hard masking (feature * 0.01) destroys visible keypoint features
when VisNet makes false negative predictions. Soft attention masking instead
reduces the INFLUENCE of occluded keypoints on others' updates while
preserving their own features intact.

Evidence: Hard masking caused hip 100%→83%, ankle 100%→33% regression
while improving arms +3%. Soft masking should preserve hip/ankle
while maintaining arm improvement.
"""
_base_ = ["./occ3l_curriculum_s1.py"]

model = dict(
    head=dict(
        soft_attention_mask=True,
        occ_attenuation=0.3,  # softer than 0.01: occluded attention is 30%, not 1%
    ),
)
