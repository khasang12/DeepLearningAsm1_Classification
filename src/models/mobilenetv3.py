"""MobileNetV3-based classifier for image classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV3Classifier(nn.Module):
    """MobileNetV3 Large / Small wrapper for classification.

    Features
    --------
    - Pretrained weights (optional)
    - Lightweight architecture optimized for mobile
    """

    SUPPORTED = {
        "mobilenetv3_large": models.mobilenet_v3_large,
        "mobilenetv3_small": models.mobilenet_v3_small,
    }

    def __init__(
        self,
        num_classes: int = 100,
        model_name: str = "mobilenetv3_large",
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        if model_name not in self.SUPPORTED:
            raise ValueError(f"Unsupported model: {model_name}. Choose from {list(self.SUPPORTED)}")

        weights = "DEFAULT" if pretrained else None
        self.model = self.SUPPORTED[model_name](weights=weights)

        # Replace the final linear layer (index 3)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_target_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM (last conv layer)."""
        return self.model.features[-1]
