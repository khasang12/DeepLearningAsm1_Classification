"""CLIP zero-shot classification using OpenCLIP."""

from __future__ import annotations

import torch
import torch.nn as nn
import open_clip


class CLIPZeroShotClassifier(nn.Module):
    """Zero-shot image classifier using CLIP.

    No training is required. The model computes cosine similarity between
    image embeddings and text embeddings of class name prompts.

    Architecture
    ------------
    CLIP Image Encoder → Image Embedding ─┐
                                          ├→ Cosine Similarity → Prediction
    CLIP Text  Encoder → Text Embedding  ─┘
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
    ):
        super().__init__()
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Freeze everything — no training
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self._text_features: torch.Tensor | None = None

    @torch.no_grad()
    def encode_text_prompts(self, prompts: list[str], device: str | torch.device = "cpu") -> None:
        """Pre-compute and cache normalized text embeddings for class names.

        Parameters
        ----------
        prompts : list[str]
            e.g. ["a photo of a cat", "a photo of a dog", ...]
        """
        tokens = self.tokenizer(prompts).to(device)
        self._text_features = self.clip_model.encode_text(tokens)
        self._text_features = self._text_features / self._text_features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Classify images via cosine similarity with cached text features.

        Parameters
        ----------
        images : (B, C, H, W) — preprocessed with CLIP transforms

        Returns
        -------
        logits : (B, num_classes) — cosine similarity scores (scaled by 100)
        """
        if self._text_features is None:
            raise RuntimeError("Call encode_text_prompts() before forward()")

        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Cosine similarity scaled by temperature
        logits = 100.0 * image_features @ self._text_features.T
        return logits

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predicted class indices and confidence scores.

        Returns (predicted_classes, probabilities).
        """
        logits = self.forward(images)
        probs = logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1)
        return preds, probs

    def get_preprocess(self):
        """Return the CLIP preprocessing transform."""
        return self.preprocess
