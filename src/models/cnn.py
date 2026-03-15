"""ResNet-based CNN classifier for image classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetClassifier(nn.Module):
    """ResNet-18 / 34 / 50 wrapper for classification.

    Features
    --------
    - Pretrained ImageNet weights (optional)
    - Configurable classification head
    - Feature extraction hook for Grad-CAM
    """

    SUPPORTED = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }

    def __init__(
        self,
        num_classes: int = 100,
        model_name: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if model_name not in self.SUPPORTED:
            raise ValueError(f"Unsupported model: {model_name}. Choose from {list(self.SUPPORTED)}")

        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = self.SUPPORTED[model_name](weights=weights)

        # Replace the final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(in_features, num_classes),
        )

        # Grad-CAM hooks
        self._features: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._extract_features(x)
        pooled = nn.functional.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        return self.head(pooled)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through backbone up to the last conv layer (layer4)."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        self._features = x  # Save for Grad-CAM
        return x

    def get_features(self) -> torch.Tensor | None:
        """Return the last feature map (for Grad-CAM)."""
        return self._features

    def get_target_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM hook registration."""
        return self.backbone.layer4[-1]
