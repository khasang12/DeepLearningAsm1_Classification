"""Training script for Task 2: Text Classification — RNN (BiLSTM) vs Transformer (DistilBERT)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from src.data.text_dataset import get_text_dataloaders
from src.engine.evaluator import Evaluator
from src.models.rnn import BiLSTMClassifier
from src.models.transformer_text import DistilBERTClassifier
from src.utils.config import load_config
from src.utils.logger import close_tb_writer, get_logger, get_tb_writer, setup_logger
from src.utils.seed import set_seed


# ──────────────────────────────────────────────────────────────────────
# Custom training loops for text models (different input formats)
# ──────────────────────────────────────────────────────────────────────

def train_rnn_epoch(
    model: BiLSTMClassifier,
    dataloader,
    optimizer,
    criterion,
    device,
    epoch: int,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """Train RNN for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"[RNN] Epoch {epoch}", leave=False)
    for padded, labels, lengths in pbar:
        padded, labels = padded.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(padded, lengths=lengths)
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=loss.item(), acc=correct / total)

    return {"train_loss": total_loss / total, "train_acc": correct / total}


@torch.no_grad()
def eval_rnn(model, dataloader, criterion, device):
    """Evaluate RNN on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    for padded, labels, lengths in dataloader:
        padded, labels = padded.to(device), labels.to(device)
        logits = model(padded, lengths=lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return {
        "val_loss": total_loss / total,
        "val_acc": correct / total,
        "preds": all_preds,
        "labels": all_labels,
    }


def train_transformer_epoch(
    model: DistilBERTClassifier,
    dataloader,
    optimizer,
    criterion,
    device,
    epoch: int,
    use_amp: bool = True,
) -> dict[str, float]:
    """Train Transformer for one epoch."""
    from torch.cuda.amp import GradScaler

    model.train()
    scaler = GradScaler(enabled=use_amp and device.type != "cpu")
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"[Transformer] Epoch {epoch}", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with autocast(enabled=use_amp and device.type != "cpu"):
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=loss.item(), acc=correct / total)

    return {"train_loss": total_loss / total, "train_acc": correct / total}


@torch.no_grad()
def eval_transformer(model, dataloader, criterion, device):
    """Evaluate Transformer on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return {
        "val_loss": total_loss / total,
        "val_acc": correct / total,
        "preds": all_preds,
        "labels": all_labels,
    }


# ──────────────────────────────────────────────────────────────────────
# Full training routine
# ──────────────────────────────────────────────────────────────────────

def train_model(
    model_type: str,
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    cfg,
    device: torch.device,
    evaluator: Evaluator,
    logger,
) -> dict:
    """Train a text model (RNN or Transformer) with early stopping."""
    criterion = nn.CrossEntropyLoss()
    lr = getattr(cfg.models, model_type, cfg).lr if hasattr(getattr(cfg.models, model_type, None), "lr") else cfg.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)

    if cfg.scheduler == "linear_warmup":
        from transformers import get_linear_schedule_with_warmup
        total_steps = len(train_loader) * cfg.epochs
        warmup = getattr(cfg, "warmup_steps", 500)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
        step_per_batch = True
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        step_per_batch = False

    tb_writer = get_tb_writer(Path(cfg.output_dir) / "logs" / model_type)
    model.to(device)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    is_rnn = model_type == "rnn"
    train_fn = train_rnn_epoch if is_rnn else train_transformer_epoch
    eval_fn = eval_rnn if is_rnn else eval_transformer

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_fn(model, train_loader, optimizer, criterion, device, epoch)
        val_metrics = eval_fn(model, val_loader, criterion, device)

        if not step_per_batch and scheduler:
            scheduler.step()

        for k in history:
            v = train_metrics.get(k) or val_metrics.get(k)
            history[k].append(v)
            tb_writer.add_scalar(f"{model_type}/{k}", v, epoch)

        logger.info(
            f"[{model_type.upper()}] Epoch {epoch}/{cfg.epochs} "
            f"| train_loss={train_metrics['train_loss']:.4f} "
            f"| train_acc={train_metrics['train_acc']:.4f} "
            f"| val_loss={val_metrics['val_loss']:.4f} "
            f"| val_acc={val_metrics['val_acc']:.4f}"
        )

        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            best_epoch = epoch
            patience_counter = 0
            ckpt_dir = Path(cfg.output_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / f"{model_type}_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                logger.info(f"[{model_type.upper()}] Early stopping at epoch {epoch}")
                break

    # Test evaluation
    logger.info(f"Evaluating {model_type.upper()} on test set...")
    test_metrics = eval_fn(model, test_loader, criterion, device)
    metrics = evaluator.compute_metrics(test_metrics["labels"], test_metrics["preds"])

    logger.info(f"\n{model_type.upper()} Test Metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    report = evaluator.classification_report_str(test_metrics["labels"], test_metrics["preds"])
    logger.info(f"\n{report}")

    evaluator.plot_confusion_matrix(
        test_metrics["labels"], test_metrics["preds"], model_name=model_type,
    )

    tb_writer.flush()

    return {
        "metrics": metrics,
        "history": history,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
    }


def main():
    parser = argparse.ArgumentParser(description="Task 2: Text Classification")
    parser.add_argument("--config", type=str, default="configs/text.yaml")
    args, extra_args = parser.parse_known_args()

    cfg = load_config(args.config, cli_args=extra_args)
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("text", log_dir=output_dir / "logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    logger.info("Loading DBpedia-14 dataset...")
    data = get_text_dataloaders(
        batch_size=cfg.batch_size,
        max_seq_len=cfg.max_seq_len,
        num_workers=cfg.num_workers,
        transformer_name=cfg.models.transformer.name,
    )

    evaluator = Evaluator(class_names=data["class_names"], output_dir=cfg.output_dir)
    results: dict[str, dict] = {}
    histories: dict[str, dict] = {}

    # ── RNN ───────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}\n Training RNN (BiLSTM)\n{'='*60}")
    rnn_model = BiLSTMClassifier(
        vocab_size=len(data["vocab"]),
        num_classes=data["num_classes"],
        embed_dim=cfg.models.rnn.embed_dim,
        hidden_dim=cfg.models.rnn.hidden_dim,
        num_layers=cfg.models.rnn.num_layers,
        dropout=cfg.models.rnn.dropout,
    )
    rnn_result = train_model(
        "rnn", rnn_model, data["rnn_train_loader"], data["rnn_val_loader"],
        data["rnn_test_loader"], cfg, device, evaluator, logger,
    )
    results["rnn"] = rnn_result["metrics"]
    histories["rnn"] = rnn_result["history"]

    # ── Transformer ───────────────────────────────────────────────────
    logger.info(f"\n{'='*60}\n Training Transformer (DistilBERT)\n{'='*60}")
    tf_model = DistilBERTClassifier(
        num_classes=data["num_classes"],
        model_name=cfg.models.transformer.name,
        pretrained=cfg.models.transformer.pretrained,
        freeze_backbone=cfg.models.transformer.freeze_backbone,
    )
    tf_result = train_model(
        "transformer", tf_model, data["transformer_train_loader"],
        data["transformer_val_loader"], data["transformer_test_loader"],
        cfg, device, evaluator, logger,
    )
    results["transformer"] = tf_result["metrics"]
    histories["transformer"] = tf_result["history"]

    # ── Comparison ───────────────────────────────────────────────────
    evaluator.compare_models(results, task_name="Text_RNN_vs_Transformer")
    evaluator.plot_training_curves(histories, task_name="Text_RNN_vs_Transformer")

    logger.info("\n" + "=" * 60)
    logger.info("FINAL COMPARISON — Text Classification")
    logger.info("=" * 60)
    for name, m in results.items():
        logger.info(f"  {name.upper()}: " + " | ".join(f"{k}={v:.4f}" for k, v in m.items()))

    close_tb_writer()
    logger.info("Done!")


if __name__ == "__main__":
    main()
