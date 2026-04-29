"""Automated hypothesis testing for illustration pose estimation.

Implements autoresearch-style loop:
  propose → train → evaluate → commit/rollback

Primary metric: OKS@75 on Bizarre Pose test set (487 images)
Baseline: 0.801 (Curriculum S2)
"""
