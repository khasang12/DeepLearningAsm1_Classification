"""MixUp and CutMix data augmentation for training."""

from __future__ import annotations

import numpy as np
import torch


def mixup(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply MixUp augmentation.

    Randomly interpolates pairs of images and their labels.

    Parameters
    ----------
    images : Tensor of shape (B, C, H, W)
    labels : Tensor of shape (B,) — integer class labels
    alpha : float
        Beta distribution parameter; 1.0 → uniform λ ∈ [0, 1].

    Returns
    -------
    mixed_images, labels_a, labels_b, lam
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed = lam * images + (1 - lam) * images[index]
    return mixed, labels, labels[index], lam


def cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply CutMix augmentation.

    Replaces a random rectangular patch of one image with a patch from another.

    Parameters
    ----------
    images : Tensor of shape (B, C, H, W)
    labels : Tensor of shape (B,)
    alpha : float
        Beta distribution parameter.

    Returns
    -------
    mixed_images, labels_a, labels_b, lam (adjusted for actual patch area)
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    _, _, H, W = images.shape

    # Sample bounding box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cy = np.random.randint(H)
    cx = np.random.randint(W)

    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    # Adjust lambda to reflect actual patch area
    lam = 1 - ((y2 - y1) * (x2 - x1)) / (H * W)
    return mixed, labels, labels[index], lam


def mixup_cutmix_criterion(
    criterion: torch.nn.Module,
    pred: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute the mixed loss for MixUp / CutMix.

    loss = λ · L(pred, labels_a) + (1 − λ) · L(pred, labels_b)
    """
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)
