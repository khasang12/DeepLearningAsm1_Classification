"""CLIP few-shot prototype classification using HuggingFace Transformers."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
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


class CLIPFewShotClassifier(nn.Module):
    """Few-shot classifier using CLIP prototype network (HuggingFace).

    Architecture
    ------------
    Frozen CLIP Image+Text → visual/text_projection → Concat [512+512=1024]
    → Prototype (mean per class) → Cosine Similarity → Prediction

    No neural network layers are trained. Classification by nearest prototype.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        pretrained: bool = True,
        num_classes: int = 10,
    ):
        super().__init__()
        self.model_name = model_name
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Freeze everything — no training
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.num_classes = num_classes
        self._prototypes: torch.Tensor | None = None  # (num_classes, embed_dim)

    def set_prototypes(self, prototypes: torch.Tensor | np.ndarray) -> None:
        """Set pre-computed class prototypes."""
        if isinstance(prototypes, np.ndarray):
            prototypes = torch.from_numpy(prototypes).float()
        self._prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)

    def load_prototypes_from_checkpoint(self, checkpoint_path: str, device: str = "cpu") -> None:
        """Load prototypes from a saved checkpoint file.

        The checkpoint is a dict saved by multimodal_pretrained.py containing
        'prototypes_normalized' (numpy array of shape (num_classes, 1024)).
        """
        data = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if not isinstance(data, dict):
            return

        # Try known keys
        protos = None
        for key in ('prototypes_normalized', 'prototypes', 'proto'):
            if key in data:
                protos = data[key]
                break

        if protos is None:
            return

        if isinstance(protos, np.ndarray):
            protos = torch.from_numpy(protos).float()
        elif isinstance(protos, torch.Tensor):
            protos = protos.float()

        self._prototypes = protos / protos.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor, text: str | None = None) -> torch.Tensor:
        """Classify using nearest prototype.

        Parameters
        ----------
        pixel_values : (B, C, H, W)
        text : optional text caption for multimodal features

        Returns
        -------
        logits : (B, num_classes)
        """
        if self._prototypes is None:
            raise RuntimeError(
                "Call set_prototypes() or load_prototypes_from_checkpoint() before forward()"
            )

        device = pixel_values.device
        prototypes = self._prototypes.to(device)

        image_features = _get_clip_image_embeds(self.clip_model, pixel_values)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if text:
            text_inputs = self.processor(
                text=[text], return_tensors="pt", padding=True, truncation=True
            )
            input_ids = text_inputs["input_ids"].to(device)
            attention_mask = text_inputs["attention_mask"].to(device)
            text_features = _get_clip_text_embeds(self.clip_model, input_ids, attention_mask)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            features = torch.cat([image_features, text_features], dim=-1)
        else:
            # Image-only: pad with zeros to match prototype dimension (1024)
            proto_dim = prototypes.shape[-1]
            img_dim = image_features.shape[-1]
            if proto_dim > img_dim:
                features = torch.cat([
                    image_features,
                    torch.zeros(image_features.shape[0], proto_dim - img_dim, device=device)
                ], dim=-1)
            else:
                features = image_features

        features = features / features.norm(dim=-1, keepdim=True)
        logits = 100.0 * features @ prototypes.T
        return logits

    def get_processor(self):
        """Return the CLIP processor for image/text preprocessing."""
        return self.processor
