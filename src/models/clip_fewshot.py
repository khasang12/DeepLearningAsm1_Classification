"""CLIP few-shot classification via linear probing on frozen embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
import open_clip
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.logger import get_logger


class CLIPFewShotClassifier(nn.Module):
    """Few-shot classifier using a linear probe on frozen CLIP image embeddings.

    Architecture
    ------------
    Frozen CLIP Image Encoder → Embedding → Linear Probe → Prediction

    The CLIP image encoder is frozen; only a lightweight linear classifier
    is trained on the K-shot support set per class.
    """

    def __init__(
        self,
        num_classes: int = 100,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
    ):
        super().__init__()
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained,
        )

        # Freeze CLIP encoder
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Determine embedding dimension
        embed_dim = self.clip_model.visual.output_dim

        # Trainable linear probe
        self.probe = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        self.logger = get_logger()

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract normalized CLIP image embeddings.

        Parameters
        ----------
        images : (B, C, H, W)

        Returns
        -------
        features : (B, embed_dim)
        """
        features = self.clip_model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass: frozen feature extraction + linear probe.

        Parameters
        ----------
        images : (B, C, H, W)

        Returns
        -------
        logits : (B, num_classes)
        """
        with torch.no_grad():
            features = self.extract_features(images)
        logits = self.probe(features)
        return logits

    def train_probe(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 50,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: str | torch.device = "cpu",
    ) -> dict[str, float]:
        """Train the linear probe end-to-end.

        Parameters
        ----------
        train_loader : DataLoader
            K-shot training data.
        test_loader : DataLoader
            Full test set for evaluation.

        Returns
        -------
        dict with final accuracy and best accuracy.
        """
        self.to(device)
        optimizer = torch.optim.AdamW(
            self.probe.parameters(), lr=lr, weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0

        for epoch in range(1, epochs + 1):
            # --- Train ---
            self.probe.train()
            total_loss = 0.0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                logits = self.forward(images)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                total += labels.size(0)

            scheduler.step()

            # --- Evaluate ---
            if epoch % 5 == 0 or epoch == epochs:
                acc = self.evaluate(test_loader, device)
                best_acc = max(best_acc, acc)
                self.logger.info(
                    f"[FewShot] Epoch {epoch}/{epochs} "
                    f"| train_loss={total_loss / total:.4f} | test_acc={acc:.4f}"
                )

        return {"final_acc": acc, "best_acc": best_acc}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, device: str | torch.device = "cpu") -> float:
        """Evaluate the probe on a test set."""
        self.probe.eval()
        correct = 0
        total = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = self.forward(images)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return correct / total

    def get_preprocess(self):
        """Return the CLIP preprocessing transform."""
        return self.preprocess
