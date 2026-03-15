"""Training script for Task 1: Image Classification — CNN (ResNet) vs ViT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from src.data.image_dataset import get_image_dataloaders
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.interpret.gradcam import GradCAM
from src.models.cnn import ResNetClassifier
from src.models.vit import ViTClassifier
from src.utils.config import load_config
from src.utils.logger import close_tb_writer, get_logger, setup_logger
from src.utils.seed import set_seed


def build_model_and_trainer(
    model_type: str,
    cfg,
    num_classes: int,
    device: torch.device,
) -> tuple[nn.Module, Trainer]:
    """Instantiate a model and its Trainer wrapper."""
    if model_type == "cnn":
        model_cfg = cfg.models.cnn
        model = ResNetClassifier(
            num_classes=num_classes,
            model_name=model_cfg.name,
            pretrained=model_cfg.pretrained,
        )
    elif model_type == "vit":
        model_cfg = cfg.models.vit
        model = ViTClassifier(
            num_classes=num_classes,
            model_name=model_cfg.name,
            pretrained=model_cfg.pretrained,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    augmentation = getattr(cfg, "augmentation", None)
    augmentation_alpha = getattr(cfg, "augmentation_alpha", 1.0)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        output_dir=cfg.output_dir,
        model_name=model_type,
        use_amp=True,
        augmentation=augmentation,
        augmentation_alpha=augmentation_alpha,
    )
    return model, trainer


def run_gradcam_demo(model, dataloader, device, output_dir: Path):
    """Generate Grad-CAM visualizations for a few test images."""
    from src.data.image_dataset import CIFAR100_CLASSES

    model.eval()
    grad_cam = GradCAM(model, model.get_target_layer())

    images, labels = next(iter(dataloader))
    images = images[:4].to(device)
    labels = labels[:4]

    heatmaps = grad_cam.generate(images)

    vis_dir = output_dir / "gradcam"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(4, len(images))):
        pred = model(images[i:i+1]).argmax(dim=1).item()
        title = f"True: {CIFAR100_CLASSES[labels[i]]} | Pred: {CIFAR100_CLASSES[pred]}"
        grad_cam.visualize(
            images[i], heatmaps[i], title=title,
            save_path=str(vis_dir / f"gradcam_sample_{i}.png"),
        )

    grad_cam.remove_hooks()


def main():
    parser = argparse.ArgumentParser(description="Task 1: Image Classification")
    parser.add_argument("--config", type=str, default="configs/image.yaml")
    args, extra_args = parser.parse_known_args()

    cfg = load_config(args.config, cli_args=extra_args)
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("image", log_dir=output_dir / "logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    logger.info("Loading CIFAR-100 dataset...")
    data = get_image_dataloaders(
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]

    evaluator = Evaluator(class_names=data["class_names"], output_dir=cfg.output_dir)
    results: dict[str, dict] = {}
    histories: dict[str, dict] = {}

    # ── Train & evaluate each model ──────────────────────────────────
    for model_type in ["cnn", "vit"]:
        logger.info(f"\n{'='*60}\n Training {model_type.upper()}\n{'='*60}")

        model, trainer = build_model_and_trainer(
            model_type, cfg, data["num_classes"], device,
        )
        result = trainer.fit(
            train_loader, val_loader,
            epochs=cfg.epochs,
            patience=cfg.early_stopping_patience,
        )
        histories[model_type] = result["history"]

        # Test evaluation
        logger.info(f"Evaluating {model_type.upper()} on test set...")
        test_metrics = trainer.evaluate(test_loader)
        metrics = evaluator.compute_metrics(test_metrics["labels"], test_metrics["preds"])
        results[model_type] = metrics

        logger.info(f"\n{model_type.upper()} Test Metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        report = evaluator.classification_report_str(test_metrics["labels"], test_metrics["preds"])
        logger.info(f"\n{report}")

        # Confusion matrix
        evaluator.plot_confusion_matrix(
            test_metrics["labels"], test_metrics["preds"],
            model_name=model_type, top_n=20,
        )

        # Grad-CAM for CNN
        if model_type == "cnn":
            logger.info("Generating Grad-CAM visualizations...")
            run_gradcam_demo(model, test_loader, device, output_dir)

    # ── Comparison ───────────────────────────────────────────────────
    evaluator.compare_models(results, task_name="Image_CNN_vs_ViT")
    evaluator.plot_training_curves(histories, task_name="Image_CNN_vs_ViT")

    logger.info("\n" + "=" * 60)
    logger.info("FINAL COMPARISON — Image Classification")
    logger.info("=" * 60)
    for name, m in results.items():
        logger.info(f"  {name.upper()}: " + " | ".join(f"{k}={v:.4f}" for k, v in m.items()))

    close_tb_writer()
    logger.info("Done! Check outputs/ for reports and visualizations.")


if __name__ == "__main__":
    main()
