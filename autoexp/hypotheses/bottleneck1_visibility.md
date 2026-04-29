# Bottleneck 1: Visibility Supervision Semantics

## Problem statement

Bizarre Pose originally had all 54,400 keypoints marked v=2 (visible).
After manual annotation, 15.2% are now v=1 (occluded).

The current `VisibilityAwareSimCC` codec with default `v1_weight=1.0` treats
occluded keypoints with the same loss weight as visible ones. This is the
WACV 2024 finding: mixing occluded into training at full weight degrades visible accuracy.

## Confirmed findings

- Curriculum learning (clean → full) helps but doesn't solve the core issue
- `OccludedRTMCCHead` with hard masking (visible*1.0, occluded*0.01) causes
  "masking visible parts" problem: hip/ankle performance decreases because
  VisNet false negatives suppress good features
- v1_weight=0.0 (visible-only training) loses information about occluded joint structure

## Hypothesis space

### H1.1: Soft v1_weight
**Hypothesis**: v1_weight=0.5 (half weight for occluded) improves OKS@75 by reducing
  noise from uncertain occluded keypoints without completely ignoring them.
**Prediction**: OKS@75 improves, OKS@50 may stay flat (robust metric).
**Config change**: VisibilityAwareSimCC with v1_weight=0.5

### H1.2: Progressive v1_weight via warmup
**Hypothesis**: Starting with v1_weight=0.2 and increasing to 1.0 over epochs
  prevents occluded noise from corrupting early training.
**Prediction**: Both OKS metrics improve slightly.
**Implementation**: Custom transform that modifies dataset metadata or codec param schedule.

### H1.3: Visibility-gated attention (soft version of OccludedRTMCCHead)
**Hypothesis**: Soft-gating keypoint features by predicted visibility score (0.0-1.0)
  rather than hard masking avoids the false-negative problem while still reducing
  occluded keypoint influence in GAU.
**Prediction**: OKS@75 improves over both OccludedRTMCCHead (hard masking) and baseline.
**Implementation**: Modify OccludedRTMCCHead to use sigmoid(vis_pred) as feature scale
  instead of binary threshold.

## Experiment priority

Start with H1.1 (simplest, most interpretable). If no gain, try H1.3.
H1.2 requires scheduler implementation, try last.
