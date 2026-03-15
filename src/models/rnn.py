"""BiLSTM classifier with self-attention pooling for text classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SelfAttentionPooling(nn.Module):
    """Compute a weighted sum of LSTM hidden states using learned attention.

    Produces a single vector from variable-length sequences.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        hidden_states : (B, T, H)
        mask : (B, T) — 1 for valid positions, 0 for padding

        Returns
        -------
        context : (B, H)
        attn_weights : (B, T) — attention distribution over positions
        """
        scores = self.attention(hidden_states).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)  # (B, T)
        context = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)  # (B, H)
        return context, attn_weights


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM with self-attention pooling.

    Architecture
    ------------
    Embedding → BiLSTM (N layers) → Self-Attention → LayerNorm → FC head

    The self-attention mechanism learns to focus on the most relevant
    tokens in the sequence, producing interpretable attention weights.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 4,
        embed_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = SelfAttentionPooling(hidden_dim * 2)  # *2 for bidirectional
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._last_attention_weights: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : (B, T) — padded token indices
        labels : unused, for API compatibility with Trainer
        lengths : (B,) — actual sequence lengths (before padding)
        """
        B, T = input_ids.shape
        embeds = self.embedding(input_ids)  # (B, T, E)

        if lengths is not None:
            # Sort by length (required for pack_padded_sequence)
            lengths_clamped = lengths.clamp(min=1)
            packed = pack_padded_sequence(
                embeds, lengths_clamped.cpu(), batch_first=True, enforce_sorted=False,
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=T)
        else:
            lstm_out, _ = self.lstm(embeds)

        # Attention pooling
        mask = (input_ids != 0).float()  # (B, T)
        context, attn_weights = self.attention(lstm_out, mask)
        self._last_attention_weights = attn_weights.detach()

        context = self.layer_norm(context)
        logits = self.head(context)
        return logits

    def get_attention_weights(self) -> torch.Tensor | None:
        """Return the last attention weights for interpretability.

        Shape: (B, T)
        """
        return self._last_attention_weights
