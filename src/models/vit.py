"""Vision Transformer (ViT) classifier using timm."""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class ViTClassifier(nn.Module):
    """ViT classifier built on the ``timm`` library.

    Features
    --------
    - Pretrained weights from timm model zoo
    - Configurable classification head
    - Attention weight extraction for interpretability
    """

    def __init__(
        self,
        num_classes: int = 100,
        model_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove default head
            drop_rate=dropout,
        )
        embed_dim = self.model.embed_dim

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(embed_dim, num_classes),
        )

        self._attention_weights: list[torch.Tensor] = []
        self._hooks: list = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(x)  # (B, N, D)
        cls_token = features[:, 0]  # CLS token
        return self.head(cls_token)

    # ------------------------------------------------------------------
    # Attention extraction
    # ------------------------------------------------------------------

    def register_attention_hooks(self) -> None:
        """Register hooks on all transformer blocks to capture attention maps."""
        self._attention_weights.clear()
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        for block in self.model.blocks:
            hook = block.attn.register_forward_hook(self._attn_hook)
            self._hooks.append(hook)

    def _attn_hook(self, module, input, output):
        """Hook function that captures attention weights during forward pass."""
        # timm's Attention module stores attn weights if we use attn_drop
        # We need to compute them manually from Q, K
        B, N, C = input[0].shape
        qkv = module.qkv(input[0]).reshape(B, N, 3, module.num_heads, C // module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        self._attention_weights.append(attn.detach().cpu())

    def get_attention_maps(self) -> list[torch.Tensor]:
        """Return captured attention maps from all layers.

        Each element has shape (B, num_heads, N, N).
        """
        return self._attention_weights

    def remove_attention_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._attention_weights.clear()
