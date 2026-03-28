"""Structural constraint losses for pose estimation.

Adds bone-length ratio consistency and bilateral symmetry as
differentiable regularization, injecting anatomical priors that
the model cannot learn from 3200 images alone.

These constraints are domain-agnostic (human skeleton proportions
apply to anime characters despite stylization) and help most when
visual features are ambiguous (clothing hiding arm contours).
"""

import torch
import torch.nn as nn

# COCO17 skeleton bone pairs (joint index pairs)
BONE_PAIRS = [
    (5, 7),   # left upper arm (shoulder -> elbow)
    (7, 9),   # left forearm (elbow -> wrist)
    (6, 8),   # right upper arm
    (8, 10),  # right forearm
    (5, 11),  # left torso (shoulder -> hip)
    (6, 12),  # right torso
    (11, 13), # left thigh (hip -> knee)
    (13, 15), # left shin (knee -> ankle)
    (12, 14), # right thigh
    (14, 16), # right shin
]

# Symmetric bone pairs (left index, right index in BONE_PAIRS)
SYMMETRIC_PAIRS = [
    (0, 2),  # left upper arm <-> right upper arm
    (1, 3),  # left forearm <-> right forearm
    (4, 5),  # left torso <-> right torso
    (6, 8),  # left thigh <-> right thigh
    (7, 9),  # left shin <-> right shin
]

# Torso bone index for normalization (mean of left/right torso)
TORSO_INDICES = (4, 5)


def soft_argmax_from_simcc(simcc_x, simcc_y):
    """Extract differentiable coordinates from SimCC logits.

    Args:
        simcc_x: (B, K, W) raw logits for x
        simcc_y: (B, K, H) raw logits for y

    Returns:
        coords: (B, K, 2) differentiable coordinates
    """
    # Softmax to get probability distributions
    px = torch.softmax(simcc_x, dim=-1)
    py = torch.softmax(simcc_y, dim=-1)

    # Weighted sum (soft argmax)
    bins_x = torch.arange(simcc_x.shape[-1], device=simcc_x.device, dtype=simcc_x.dtype)
    bins_y = torch.arange(simcc_y.shape[-1], device=simcc_y.device, dtype=simcc_y.dtype)

    x = (px * bins_x).sum(dim=-1)  # (B, K)
    y = (py * bins_y).sum(dim=-1)  # (B, K)

    return torch.stack([x, y], dim=-1)  # (B, K, 2)


def bone_lengths(coords, bone_pairs):
    """Compute bone lengths from coordinates.

    Args:
        coords: (B, K, 2)
        bone_pairs: list of (i, j) index pairs

    Returns:
        lengths: (B, num_bones)
    """
    lengths = []
    for i, j in bone_pairs:
        d = torch.norm(coords[:, i] - coords[:, j], dim=-1)
        lengths.append(d)
    return torch.stack(lengths, dim=-1)


def symmetry_loss(bone_lens):
    """Penalize asymmetry between left and right bones.

    Human bodies are approximately bilaterally symmetric.
    This constraint is domain-agnostic.
    """
    loss = 0.0
    count = 0
    for li, ri in SYMMETRIC_PAIRS:
        diff = (bone_lens[:, li] - bone_lens[:, ri]).abs()
        # Normalize by mean length to make scale-invariant
        mean_len = (bone_lens[:, li] + bone_lens[:, ri]) / 2 + 1e-6
        loss = loss + (diff / mean_len).mean()
        count += 1
    return loss / count


def ratio_consistency_loss(bone_lens):
    """Penalize bone ratios that deviate from anatomical priors.

    Uses torso length as normalization reference.
    Arm bones should be proportional to torso.
    """
    # Torso length (mean of left and right shoulder-hip)
    torso = (bone_lens[:, TORSO_INDICES[0]] + bone_lens[:, TORSO_INDICES[1]]) / 2 + 1e-6

    # Normalize all bones by torso
    ratios = bone_lens / torso.unsqueeze(-1)

    # Variance of ratios across the batch should be low
    # (same person's proportions shouldn't change drastically with augmentation)
    ratio_var = ratios.var(dim=0).mean()

    return ratio_var


def structural_loss(simcc_x, simcc_y, target_weight, alpha_sym=0.1, alpha_ratio=0.05):
    """Combined structural loss.

    Args:
        simcc_x: (B, K, W) predicted SimCC x logits
        simcc_y: (B, K, H) predicted SimCC y logits
        target_weight: (B, K, 1) visibility weights
        alpha_sym: weight for symmetry loss
        alpha_ratio: weight for ratio consistency loss
    """
    coords = soft_argmax_from_simcc(simcc_x, simcc_y)

    # Only compute on samples where relevant keypoints are visible
    # Mask: both endpoints of each bone must be visible
    weight_flat = target_weight.squeeze(-1)  # (B, K)
    bone_mask = []
    for i, j in BONE_PAIRS:
        visible = (weight_flat[:, i] > 0.5) & (weight_flat[:, j] > 0.5)
        bone_mask.append(visible.float())
    bone_mask = torch.stack(bone_mask, dim=-1)  # (B, num_bones)

    bone_lens = bone_lengths(coords, BONE_PAIRS)

    # Apply mask (set invisible bone lengths to 0 so they don't affect loss)
    bone_lens_masked = bone_lens * bone_mask

    # Only compute if we have enough visible bones
    if bone_mask.sum() < 2:
        return torch.tensor(0.0, device=simcc_x.device)

    sym = symmetry_loss(bone_lens_masked)
    ratio = ratio_consistency_loss(bone_lens_masked)

    return alpha_sym * sym + alpha_ratio * ratio
