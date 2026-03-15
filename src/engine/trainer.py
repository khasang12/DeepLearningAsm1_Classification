"""Generic training loop with early stopping, mixed-precision, and checkpointing."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.augmentations import cutmix, mixup, mixup_cutmix_criterion
from src.utils.logger import get_logger, get_tb_writer


class Trainer:
    """A reusable training engine.

    Supports:
    - Mixed-precision training (AMP)
    - Gradient clipping
    - Learning-rate scheduling
    - Early stopping
    - MixUp / CutMix augmentation
    - TensorBoard logging
    - Checkpoint save/load
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Any | None = None,
        device: str | torch.device = "cpu",
        output_dir: str | Path = "outputs",
        model_name: str = "model",
        use_amp: bool = True,
        grad_clip: float = 1.0,
        augmentation: str | None = None,  # "mixup", "cutmix", or None
        augmentation_alpha: float = 1.0,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.use_amp = use_amp and device != "cpu"
        self.grad_clip = grad_clip
        self.augmentation = augmentation
        self.augmentation_alpha = augmentation_alpha

        self.scaler = GradScaler(enabled=self.use_amp)
        self.logger = get_logger()
        self.tb_writer = get_tb_writer(self.output_dir / "logs" / model_name)

        # Checkpoint directory
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_one_epoch(self, dataloader: DataLoader, epoch: int) -> dict[str, float]:
        """Train for a single epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"[{self.model_name}] Epoch {epoch}", leave=False)
        for batch in pbar:
            images, labels = self._unpack_batch(batch)
            images, labels = images.to(self.device), labels.to(self.device)

            # Optional augmentation
            use_mixed_loss = False
            if self.augmentation == "mixup":
                images, labels_a, labels_b, lam = mixup(images, labels, self.augmentation_alpha)
                use_mixed_loss = True
            elif self.augmentation == "cutmix":
                images, labels_a, labels_b, lam = cutmix(images, labels, self.augmentation_alpha)
                use_mixed_loss = True

            self.optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                if use_mixed_loss:
                    loss = mixup_cutmix_criterion(self.criterion, logits, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=loss.item(), acc=correct / total)

        avg_loss = total_loss / total
        accuracy = correct / total
        return {"train_loss": avg_loss, "train_acc": accuracy}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate the model on a validation/test set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds: list[int] = []
        all_labels: list[int] = []

        for batch in dataloader:
            images, labels = self._unpack_batch(batch)
            images, labels = images.to(self.device), labels.to(self.device)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / total
        accuracy = correct / total
        return {
            "val_loss": avg_loss,
            "val_acc": accuracy,
            "preds": all_preds,
            "labels": all_labels,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 7,
    ) -> dict[str, Any]:
        """Full training loop with early stopping.

        Returns
        -------
        dict with keys: best_val_acc, best_epoch, history
        """
        best_val_acc = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        self.logger.info(
            f"Starting training for {self.model_name} | "
            f"epochs={epochs}, patience={patience}, device={self.device}"
        )

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_one_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.time() - t0

            # Log
            for k in ("train_loss", "train_acc", "val_loss", "val_acc"):
                v = train_metrics.get(k) or val_metrics.get(k)
                history[k].append(v)
                self.tb_writer.add_scalar(f"{self.model_name}/{k}", v, epoch)

            self.logger.info(
                f"[{self.model_name}] Epoch {epoch}/{epochs} "
                f"| train_loss={train_metrics['train_loss']:.4f} "
                f"| train_acc={train_metrics['train_acc']:.4f} "
                f"| val_loss={val_metrics['val_loss']:.4f} "
                f"| val_acc={val_metrics['val_acc']:.4f} "
                f"| time={elapsed:.1f}s"
            )

            # Early stopping
            if val_metrics["val_acc"] > best_val_acc:
                best_val_acc = val_metrics["val_acc"]
                best_epoch = epoch
                epochs_no_improve = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    self.logger.info(
                        f"[{self.model_name}] Early stopping at epoch {epoch} "
                        f"(best={best_val_acc:.4f} at epoch {best_epoch})"
                    )
                    break

        self.tb_writer.flush()
        return {
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "history": history,
            "final_val_metrics": val_metrics,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> Path:
        """Save model + optimizer state."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        path = self.ckpt_dir / f"{self.model_name}_epoch{epoch}.pt"
        torch.save(state, path)
        if is_best:
            best_path = self.ckpt_dir / f"{self.model_name}_best.pt"
            torch.save(state, best_path)
        return path

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a checkpoint and return the epoch number."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["epoch"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_batch(batch):
        """Support both tuple batches and dict batches."""
        if isinstance(batch, (list, tuple)):
            return batch[0], batch[1]
        return batch["pixel_values"], batch["labels"]
