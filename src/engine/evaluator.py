"""Evaluation utilities — metrics, confusion matrices, and comparison reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.logger import get_logger


class Evaluator:
    """Compute classification metrics and generate visual reports."""

    def __init__(
        self,
        class_names: list[str] | None = None,
        output_dir: str | Path = "outputs",
    ):
        self.class_names = class_names
        self.output_dir = Path(output_dir) / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()

    def compute_metrics(
        self,
        y_true: list[int],
        y_pred: list[int],
    ) -> dict[str, float]:
        """Compute accuracy, precision, recall, and F1 (macro)."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    def classification_report_str(
        self,
        y_true: list[int],
        y_pred: list[int],
    ) -> str:
        """Generate a full per-class classification report string."""
        target_names = self.class_names if self.class_names else None
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

    def plot_confusion_matrix(
        self,
        y_true: list[int],
        y_pred: list[int],
        model_name: str = "model",
        normalize: bool = True,
        top_n: int | None = 20,
    ) -> Path:
        """Save a confusion matrix heatmap as PNG.

        Parameters
        ----------
        top_n : int | None
            If the number of classes exceeds *top_n*, only the top-N most
            frequent classes are shown to keep the plot readable.
        """
        labels = sorted(set(y_true))

        # Optionally limit to top-N classes for readability
        if top_n and len(labels) > top_n:
            from collections import Counter
            counts = Counter(y_true)
            labels = [l for l, _ in counts.most_common(top_n)]
            mask = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t in labels]
            y_true = [y_true[i] for i in mask]
            y_pred = [y_pred[i] for i in mask]

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if normalize:
            cm = cm.astype(np.float64)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # avoid division by zero
            cm = cm / row_sums

        display_labels = (
            [self.class_names[i] for i in labels]
            if self.class_names and max(labels) < len(self.class_names)
            else [str(l) for l in labels]
        )

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.4), max(6, len(labels) * 0.35)))
        sns.heatmap(
            cm,
            annot=len(labels) <= 25,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=display_labels,
            yticklabels=display_labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {model_name}")
        plt.tight_layout()

        save_path = self.output_dir / f"confusion_matrix_{model_name}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        self.logger.info(f"Confusion matrix saved to {save_path}")
        return save_path

    def compare_models(
        self,
        results: dict[str, dict[str, float]],
        task_name: str = "comparison",
    ) -> Path:
        """Generate a comparison bar chart for multiple models.

        Parameters
        ----------
        results : dict[str, dict[str, float]]
            Mapping from model name → metrics dict (accuracy, f1_macro, etc.).
        """
        model_names = list(results.keys())
        metric_names = list(next(iter(results.values())).keys())

        x = np.arange(len(metric_names))
        width = 0.8 / len(model_names)

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, name in enumerate(model_names):
            values = [results[name].get(m, 0) for m in metric_names]
            offset = (i - len(model_names) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=name)
            # Add value labels on bars
            for bar, v in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_ylabel("Score")
        ax.set_title(f"Model Comparison — {task_name}")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.set_ylim(0, 1.1)
        plt.tight_layout()

        save_path = self.output_dir / f"comparison_{task_name}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        self.logger.info(f"Comparison chart saved to {save_path}")
        return save_path

    def plot_training_curves(
        self,
        histories: dict[str, dict[str, list[float]]],
        task_name: str = "training",
    ) -> Path:
        """Plot training & validation loss/accuracy curves for multiple models."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for name, history in histories.items():
            epochs = range(1, len(history["train_loss"]) + 1)
            axes[0].plot(epochs, history["train_loss"], label=f"{name} train")
            axes[0].plot(epochs, history["val_loss"], "--", label=f"{name} val")
            axes[1].plot(epochs, history["train_acc"], label=f"{name} train")
            axes[1].plot(epochs, history["val_acc"], "--", label=f"{name} val")

        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"Training Curves — {task_name}")
        plt.tight_layout()

        save_path = self.output_dir / f"training_curves_{task_name}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        self.logger.info(f"Training curves saved to {save_path}")
        return save_path
