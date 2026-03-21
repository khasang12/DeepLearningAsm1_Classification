"""BERT-based text classifier with frozen backbone (linear probing)."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification


class BERTClassifier(nn.Module):
    """BERT with frozen backbone for text classification (linear probing).

    Architecture
    ------------
    BERT (bert-base-uncased, frozen) → [CLS] → Linear(768, num_classes)

    Follows the same pattern as the Colab notebook (text_pretrained.py):
    - Uses HuggingFace BertForSequenceClassification
    - Entire BERT backbone is FROZEN
    - Only the classifier head (768 → num_classes) is trained
    - LR=1e-2 (linear probing, not fine-tuning)
    """

    def __init__(
        self,
        num_classes: int = 14,
        model_name: str = "bert-base-uncased",
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.model_name = model_name

        if pretrained:
            self.model = BertForSequenceClassification.from_pretrained(
                model_name, num_labels=num_classes
            )
        else:
            from transformers import BertConfig
            config = BertConfig.from_pretrained(model_name, num_labels=num_classes)
            self.model = BertForSequenceClassification(config)

        # Freeze BERT backbone, keep only classifier head trainable
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : (B, T)
        attention_mask : (B, T)

        Returns
        -------
        logits : (B, num_classes)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits
