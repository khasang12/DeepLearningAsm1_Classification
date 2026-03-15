"""DistilBERT-based text classifier using HuggingFace Transformers."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class DistilBERTClassifier(nn.Module):
    """DistilBERT fine-tuning wrapper for text classification.

    Architecture
    ------------
    DistilBERT → [CLS] token → LayerNorm → Dropout → FC head

    Supports:
    - Full fine-tuning (all params trainable)
    - Frozen backbone (only head is trained)
    - Attention weight extraction for interpretability
    """

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = "distilbert-base-uncased",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name) if pretrained else AutoModel.from_config(
            AutoModel.from_pretrained(model_name).config
        )
        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

        self._last_attentions: tuple | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : (B, T)
        attention_mask : (B, T)
        labels : unused, for API compatibility

        Returns
        -------
        logits : (B, num_classes)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        # CLS token representation
        cls_output = outputs.last_hidden_state[:, 0]  # (B, H)
        self._last_attentions = outputs.attentions  # tuple of (B, heads, T, T)

        logits = self.head(cls_output)
        return logits

    def get_attention_maps(self) -> tuple | None:
        """Return attention maps from all layers.

        Each element: (B, num_heads, T, T)
        """
        return self._last_attentions

    def get_cls_attention(self, layer: int = -1) -> torch.Tensor | None:
        """Get attention weights from CLS token to all other tokens.

        Returns shape (B, num_heads, T).
        """
        if self._last_attentions is None:
            return None
        attn = self._last_attentions[layer]  # (B, heads, T, T)
        return attn[:, :, 0, :]  # CLS row
