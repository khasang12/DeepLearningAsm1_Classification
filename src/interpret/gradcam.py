"""Grad-CAM implementation for CNN interpretability."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM).

    Generates visual explanations for CNN predictions by highlighting
    image regions that are important for the predicted class.

    References
    ----------
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Parameters
        ----------
        model : nn.Module
            The full model (e.g., ResNetClassifier).
        target_layer : nn.Module
            The convolutional layer to compute Grad-CAM on
            (e.g., model.get_target_layer()).
        """
        self.model = model
        self.target_layer = target_layer

        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        images: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmaps.

        Parameters
        ----------
        images : (B, C, H, W)
        target_class : int | None
            Class index to generate CAM for. If None, uses predicted class.

        Returns
        -------
        heatmaps : np.ndarray of shape (B, H, W) in [0, 1]
        """
        self.model.eval()
        images.requires_grad_(True)

        # Forward
        logits = self.model(images)

        if target_class is None:
            target_class = logits.argmax(dim=1)

        # Create one-hot target
        one_hot = torch.zeros_like(logits)
        if isinstance(target_class, int):
            one_hot[:, target_class] = 1
        else:
            one_hot.scatter_(1, target_class.unsqueeze(1), 1)

        # Backward
        self.model.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=True)

        # Compute Grad-CAM
        gradients = self._gradients  # (B, C, h, w)
        activations = self._activations  # (B, C, h, w)

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * activations).sum(dim=1)  # (B, h, w)
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=images.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Normalize per sample
        B = cam.shape[0]
        cam = cam.view(B, -1)
        cam_min = cam.min(dim=1, keepdim=True).values
        cam_max = cam.max(dim=1, keepdim=True).values
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam = cam.view(B, images.shape[2], images.shape[3])

        return cam.cpu().numpy()

    def visualize(
        self,
        image: torch.Tensor,
        heatmap: np.ndarray,
        title: str = "Grad-CAM",
        save_path: str | None = None,
        alpha: float = 0.4,
    ) -> plt.Figure:
        """Overlay Grad-CAM heatmap on the original image.

        Parameters
        ----------
        image : (C, H, W) — original image tensor (normalized)
        heatmap : (H, W) — Grad-CAM heatmap in [0, 1]
        """
        # Denormalize image for display
        img = image.detach().cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.5071, 0.4867, 0.4408])
        std = np.array([0.2675, 0.2565, 0.2761])
        img = img * std + mean
        img = np.clip(img, 0, 1)

        # Create colored heatmap
        colored_heatmap = cm.jet(heatmap)[:, :, :3]

        # Overlay
        overlay = alpha * colored_heatmap + (1 - alpha) * img
        overlay = np.clip(overlay, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def remove_hooks(self) -> None:
        """Remove the registered hooks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()
