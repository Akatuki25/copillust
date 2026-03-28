"""Verify whether DINOv2 can recognize body part structure in anime illustrations.

Extracts DINOv2 patch features from mydata images, computes patch-to-patch
affinity, and applies spectral decomposition to get unsupervised part maps.
If DINOv2 understands body structure in anime, the eigenvectors should
correspond to distinct body parts (head, torso, arms, legs).

Outputs visualization images showing:
1. The original image
2. Top eigenvector maps (each highlighting a different body region)
3. Whether arm regions are distinguishable from torso
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

def main():
    mydata_dir = Path("mydata")
    output_dir = Path("experiments/eval/dinov2_parts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load DINOv2
    print("Loading DINOv2 ViT-B/14...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    model.eval()

    # DINOv2 ViT-B/14: patch_size=14, embed_dim=768
    patch_size = 14

    # Process each mydata image
    images = sorted(
        list(mydata_dir.rglob("*.jpeg")) +
        list(mydata_dir.rglob("*.png")) +
        list(mydata_dir.rglob("*.jpg"))
    )

    for img_path in images:
        if "annotations" in str(img_path):
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]

        # Resize to be divisible by patch_size
        h_new = (h_orig // patch_size) * patch_size
        w_new = (w_orig // patch_size) * patch_size
        img_resized = cv2.resize(img_rgb, (w_new, h_new))

        # Normalize (ImageNet stats)
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.unsqueeze(0)

        # Extract patch features
        with torch.no_grad():
            features = model.forward_features(img_tensor)
            patch_tokens = features["x_norm_patchtokens"]  # (1, N, 768)

        n_patches = patch_tokens.shape[1]
        h_patches = h_new // patch_size
        w_patches = w_new // patch_size

        # Compute affinity matrix (cosine similarity between patches)
        feats = patch_tokens.squeeze(0)  # (N, 768)
        feats_norm = F.normalize(feats, dim=-1)
        affinity = feats_norm @ feats_norm.T  # (N, N)

        # Spectral decomposition (Laplacian eigenvectors)
        # D = degree matrix, L = D - A (unnormalized Laplacian)
        affinity_np = affinity.cpu().numpy()
        # Threshold to keep only strong connections
        affinity_np[affinity_np < 0.5] = 0
        degree = affinity_np.sum(axis=1)
        D = np.diag(degree)
        L = D - affinity_np

        # Solve generalized eigenvalue problem
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            print(f"  Eigendecomposition failed for {img_path.name}")
            continue

        # Skip first eigenvector (constant), take next 5
        n_components = min(6, eigenvectors.shape[1])

        # Create visualization
        panels = []

        # Original image (resized to match patch grid)
        vis_h = 300
        vis_scale = vis_h / h_new
        vis_w = int(w_new * vis_scale)
        orig_vis = cv2.resize(img_bgr, (vis_w, vis_h))
        panels.append(orig_vis)

        # Eigenvector maps
        for k in range(1, n_components):
            ev = eigenvectors[:, k].reshape(h_patches, w_patches)

            # Normalize to 0-255
            ev_norm = (ev - ev.min()) / (ev.max() - ev.min() + 1e-8)
            ev_img = (ev_norm * 255).astype(np.uint8)

            # Upscale to visualization size
            ev_resized = cv2.resize(ev_img, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST)

            # Apply colormap
            ev_colored = cv2.applyColorMap(ev_resized, cv2.COLORMAP_JET)

            # Overlay on original
            overlay = cv2.addWeighted(orig_vis, 0.4, ev_colored, 0.6, 0)

            # Label
            cv2.putText(overlay, f"EV{k}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            panels.append(overlay)

        # Concatenate horizontally
        result = np.hstack(panels)

        cat = img_path.parent.name
        out_path = output_dir / f"{cat}_{img_path.stem}_dinov2.jpg"
        cv2.imwrite(str(out_path), result)
        print(f"  {cat}/{img_path.name}")

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
