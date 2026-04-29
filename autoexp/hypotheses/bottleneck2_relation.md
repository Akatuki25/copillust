# Bottleneck 2: GAU Relation Prior

## Problem statement

RTMCCHead uses GAU (Gated Attention Unit) with 17 keypoint tokens to model
joint relationships. The GAU weights encode relationships learned from
COCO + HumanArt data. For anime/illustration with deformed proportions
(2-3 head ratio, extremely long/short limbs, chibi body type), these
priors from natural images may be actively harmful.

## Confirmed findings

- HumanArt pretrained significantly better than COCO pretrained (+0.16 OKS@50)
  → GAU relation prior IS the right abstraction, just needs better initialization
- Bizarre Pose fine-tune further adapts the GAU to anime
- mydata extreme poses (inverted body, dense lineart) still fail

## Hypothesis space

### H2.1: Reduced GAU hidden_dims
**Hypothesis**: Reducing GAU hidden_dims from 256 to 128 reduces over-reliance
  on COCO-learned joint correlations, forcing more localized predictions.
**Prediction**: Improved OKS@75 for extreme/deformed poses. May reduce OKS@50 slightly.
**Config change**: gau_cfg.hidden_dims=128

### H2.2: GAU dropout on attention weights
**Hypothesis**: Adding dropout_rate=0.1 to GAU forces less reliance on fixed
  pairwise joint priors, improving generalization to unusual poses.
**Prediction**: Small improvement, especially on tail poses.
**Config change**: gau_cfg.dropout_rate=0.1

### H2.3: Higher expansion_factor
**Hypothesis**: expansion_factor=4 (vs current 2) gives GAU more capacity to
  learn illustration-specific relationships without changing attention structure.
**Prediction**: Marginal improvement. Risk: overfitting on 3200 images.
**Config change**: gau_cfg.expansion_factor=4

### H2.4: Separate GAU for face vs body keypoints
**Hypothesis**: Face keypoints (0-4: nose/eyes/ears) have very different relationship
  patterns in illustrations vs body keypoints. Using separate GAU blocks per group
  reduces cross-contamination.
**Prediction**: Improved face keypoint localization, especially for occluded face.
**Implementation**: Custom head with 2-group GAU.

## Experiment priority

H2.1 and H2.2 are the fastest to test (config-only changes).
H2.4 is highest potential but requires new head implementation.
