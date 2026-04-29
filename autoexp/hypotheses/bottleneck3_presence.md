# Bottleneck 3: Tail Failure / Presence

## Problem statement

RTMPose (top-down, SimCC) assumes the crop contains the full body.
For illustration, this assumption breaks frequently:
- Bust-up shots (knees/ankles out of frame)
- Extreme crop from artist composition
- Partial visibility at image border

When a keypoint is out-of-frame, RTMPose still predicts a coordinate.
The prediction is typically near the crop boundary but wrong.
These "tail" predictions (confident but wrong) drag down OKS@50.

Confidence threshold experiment: filtering low-confidence predictions
improved OKS@50 from 0.892 → 0.934, confirming presence as a bottleneck.

## Confirmed findings

- OKS@50 vs OKS@75 gap: our model has relatively better precision (OKS@75)
  but worse recall coverage (OKS@50). Tail failures reduce the former.
- crop-aug experiment (P1 crop-augment) attempted ProbPose approach, result TBD.

## Hypothesis space

### H3.1: Presence auxiliary head (simple)
**Hypothesis**: Adding a per-keypoint binary presence head (sigmoid output)
  that learns to predict "is this keypoint inside the crop?" allows suppressing
  out-of-frame predictions during inference.
**Prediction**: OKS@50 improves significantly (currently 0.892, potential +0.006).
  OKS@75 may stay flat or improve slightly.
**Implementation**: Thin MLP on backbone features → 17 binary logits.
  Loss: BCE with presence labels derived from GT keypoints (v=0 near border → not present).

### H3.2: Boundary-aware loss weight
**Hypothesis**: Downweighting keypoints predicted near crop boundaries during
  training reduces the model's tendency to place OOF keypoints at edges.
**Prediction**: Reduced tail failures.
**Implementation**: Transform that sets lower weight for keypoints within 10px of crop edge.

### H3.3: Random crop augmentation (ProbPose-style)
**Hypothesis**: Training with random edge blackouts teaches the model that
  some keypoints may be absent from the crop.
**Prediction**: Both OKS metrics improve on bust-up/partial-body images.
**Implementation**: Use existing `random_edges_blackout.py` transform, tune coverage.

## Experiment priority

H3.3 is already partially implemented (random_edges_blackout.py exists).
Start with H3.3 (tune parameters), then H3.1 (new head, higher potential).
H3.2 is a simple augmentation fallback.
