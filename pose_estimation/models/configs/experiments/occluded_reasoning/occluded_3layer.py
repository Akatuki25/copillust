"""Occluded reasoning with 3-layer GAU (WACV 2024 paper uses 3 transformer layers).

1-layer GAU showed limited improvement because inter-keypoint reasoning
for occluded joints requires multiple rounds of message passing.
Example: inferring left elbow hidden behind torso requires
  Layer 1: shoulder attends to visible features
  Layer 2: elbow attends to shoulder's updated representation
  Layer 3: refinement with full context

WACV 2024 uses 3 transformer layers with 512 dim.
"""
_base_ = ["./occluded_reasoning.py"]

model = dict(
    head=dict(
        num_gau_layers=3,
    ),
)
