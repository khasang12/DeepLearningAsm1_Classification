"""Attention visualization for ViT and text models."""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt


def visualize_vit_attention(
    attention_maps: list[torch.Tensor],
    image: torch.Tensor | None = None,
    patch_size: int = 16,
    image_size: int = 224,
    layer: int = -1,
    head: int | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Visualize ViT attention from CLS token to image patches.

    Parameters
    ----------
    attention_maps : list[Tensor]
        List of attention maps from each layer, shape (B, heads, N, N).
    image : Tensor | None
        Original image (C, H, W) for overlay (normalized).
    patch_size : int
        ViT patch size.
    image_size : int
        Input image size.
    layer : int
        Which layer's attention to visualize (-1 for last).
    head : int | None
        Specific head to show. If None, averages across heads.
    """
    attn = attention_maps[layer][0]  # First sample: (heads, N, N)

    if head is not None:
        attn = attn[head]  # (N, N)
    else:
        attn = attn.mean(dim=0)  # Average across heads: (N, N)

    # CLS token attention to patches (skip CLS position)
    cls_attn = attn[0, 1:]  # (num_patches,)

    num_patches_per_side = image_size // patch_size
    cls_attn = cls_attn.reshape(num_patches_per_side, num_patches_per_side).numpy()

    # Upsample
    from scipy.ndimage import zoom
    cls_attn = zoom(cls_attn, image_size / num_patches_per_side, order=1)

    # Normalize
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

    if image is not None:
        # Denormalize
        img = image.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.5071, 0.4867, 0.4408])
        std = np.array([0.2675, 0.2565, 0.2761])
        img = np.clip(img * std + mean, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(cls_attn, cmap="viridis")
        axes[1].set_title(f"CLS Attention (L{layer})")
        axes[1].axis("off")

        # Overlay
        overlay = 0.4 * plt.cm.viridis(cls_attn)[:, :, :3] + 0.6 * img
        axes[2].imshow(np.clip(overlay, 0, 1))
        axes[2].set_title("Overlay")
        axes[2].axis("off")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(cls_attn, cmap="viridis")
        ax.set_title(f"CLS Attention (Layer {layer})")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def visualize_text_attention(
    attention_weights: torch.Tensor,
    tokens: list[str],
    top_k: int = 30,
    save_path: str | None = None,
) -> plt.Figure:
    """Visualize attention weights over text tokens as a horizontal bar chart.

    Parameters
    ----------
    attention_weights : Tensor of shape (T,)
        Attention weights from BiLSTM self-attention.
    tokens : list[str]
        Corresponding token strings.
    top_k : int
        Show only the top-K tokens by attention weight.
    """
    weights = attention_weights.cpu().numpy()

    # Truncate to actual token length
    n = min(len(tokens), len(weights))
    tokens = tokens[:n]
    weights = weights[:n]

    # Get top-K
    if n > top_k:
        indices = np.argsort(weights)[-top_k:]
        tokens = [tokens[i] for i in indices]
        weights = weights[indices]

    # Sort by weight for display
    order = np.argsort(weights)
    tokens = [tokens[i] for i in order]
    weights = weights[order]

    fig, ax = plt.subplots(figsize=(8, max(4, len(tokens) * 0.3)))
    colors = plt.cm.YlOrRd(weights / (weights.max() + 1e-8))
    ax.barh(range(len(tokens)), weights, color=colors)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=9)
    ax.set_xlabel("Attention Weight")
    ax.set_title("Token Attention Distribution")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
