"""CLIP zero-shot classification using HuggingFace Transformers."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


def _get_clip_image_embeds(clip_model, pixel_values):
    """Extract 512-d image embeddings using explicit projection (transformers 5.x safe)."""
    vision_outputs = clip_model.vision_model(pixel_values=pixel_values)
    image_embeds = clip_model.visual_projection(vision_outputs.pooler_output)
    return image_embeds


def _get_clip_text_embeds(clip_model, input_ids, attention_mask):
    """Extract 512-d text embeddings using explicit projection (transformers 5.x safe)."""
    text_outputs = clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    text_embeds = clip_model.text_projection(text_outputs.pooler_output)
    return text_embeds


class CLIPZeroShotClassifier(nn.Module):
    """Zero-shot image classifier using CLIP (HuggingFace).

    Architecture
    ------------
    CLIP Image Encoder → visual_projection → 512-d ─┐
                                                      ├→ Cosine Sim → Prediction
    CLIP Text  Encoder → text_projection → 512-d   ─┘

    No training required. Classifies by computing cosine similarity between
    image embeddings and text prompt embeddings ("a photo of a [class]").
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        pretrained: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Freeze everything — no training
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self._text_features: torch.Tensor | None = None

    @torch.no_grad()
    def encode_text_prompts(self, prompts: list[str], device: str | torch.device = "cpu") -> None:
        """Pre-compute and cache normalized text embeddings for class names."""
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        text_features = _get_clip_text_embeds(self.clip_model, input_ids, attention_mask)
        self._text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Classify images via cosine similarity with cached text features.

        Parameters
        ----------
        pixel_values : (B, C, H, W)

        Returns
        -------
        logits : (B, num_classes) — cosine similarity scores (scaled by 100)
        """
        if self._text_features is None:
            raise RuntimeError("Call encode_text_prompts() before forward()")

        image_features = _get_clip_image_embeds(self.clip_model, pixel_values)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ self._text_features.T
        return logits

    @torch.no_grad()
    def predict(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predicted class indices and confidence scores."""
        logits = self.forward(pixel_values)
        probs = logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1)
        return preds, probs

    def get_processor(self):
        """Return the CLIP processor for image/text preprocessing."""
        return self.processor
