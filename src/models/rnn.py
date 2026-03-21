"""BiLSTM classifier with frozen GloVe embeddings for text classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """BiLSTM with frozen GloVe embeddings + mean pooling.

    Architecture
    ------------
    Embedding (GloVe, frozen) → Dropout → BiLSTM (2 layers) → Mean Pool → FC head

    Follows the same pattern as the Colab notebook (text_pretrained.py):
    - GloVe 100d embeddings, FROZEN
    - BiLSTM hidden=128, 2 layers, bidirectional
    - Mean pooling over sequence → FC(256→128) → ReLU → FC(128→num_classes)
    """

    def __init__(
        self,
        num_classes: int = 14,
        vocab_size: int = 50000,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        pretrained_embeddings=None,
        freeze_embeddings: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Load pretrained GloVe embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(
                torch.FloatTensor(pretrained_embeddings)
            )

        # Freeze embedding layer (GloVe is pretrained, not trained)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : (B, T) — padded token indices

        Returns
        -------
        logits : (B, num_classes)
        """
        embedded = self.embedding(input_ids)        # (B, T, embed_dim)
        embedded = self.dropout(embedded)

        lstm_out, _ = self.lstm(embedded)            # (B, T, 2*hidden_dim)

        pooled = lstm_out.mean(dim=1)                # (B, 2*hidden_dim)

        out = self.dropout(pooled)
        out = self.fc1(out)                          # (B, hidden_dim)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)                          # (B, num_classes)
        return out
