"""Training script for Task 3: Multimodal Classification — Zero-shot vs Few-shot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm

from src.data.multimodal_dataset import get_multimodal_dataloaders
from src.engine.evaluator import Evaluator
from src.models.clip_fewshot import CLIPFewShotClassifier
from src.models.clip_zeroshot import CLIPZeroShotClassifier
from src.utils.config import load_config
from src.utils.logger import close_tb_writer, get_logger, setup_logger
from src.utils.seed import set_seed


@torch.no_grad()
def evaluate_zero_shot(
    model: CLIPZeroShotClassifier,
    test_loader,
    device: torch.device,
) -> dict:
    """Evaluate zero-shot CLIP classification."""
    all_preds, all_labels = [], []

    for images, labels in tqdm(test_loader, desc="[ZeroShot] Evaluating"):
        images = images.to(device)
        preds, _ = model.predict(images)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

    return {"preds": all_preds, "labels": all_labels}


@torch.no_grad()
def evaluate_few_shot(
    model: CLIPFewShotClassifier,
    test_loader,
    device: torch.device,
) -> dict:
    """Evaluate few-shot CLIP probe."""
    model.probe.eval()
    all_preds, all_labels = [], []

    for images, labels in tqdm(test_loader, desc="[FewShot] Evaluating"):
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

    return {"preds": all_preds, "labels": all_labels}


def main():
    parser = argparse.ArgumentParser(description="Task 3: Multimodal Classification")
    parser.add_argument("--config", type=str, default="configs/multimodal.yaml")
    args, extra_args = parser.parse_known_args()

    cfg = load_config(args.config, cli_args=extra_args)
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("multimodal", log_dir=output_dir / "logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Zero-shot model (also provides the preprocess function) ───────
    logger.info("Loading CLIP model for zero-shot classification...")
    zs_model = CLIPZeroShotClassifier(
        model_name=cfg.clip_model,
        pretrained=cfg.clip_pretrained,
    )
    zs_model.to(device)

    # ── Data ──────────────────────────────────────────────────────────
    logger.info("Loading CIFAR-100 for multimodal classification...")
    clip_preprocess = zs_model.get_preprocess()
    few_shot_k = cfg.few_shot_k if isinstance(cfg.few_shot_k, list) else [cfg.few_shot_k]

    data = get_multimodal_dataloaders(
        clip_preprocess=clip_preprocess,
        few_shot_k=few_shot_k,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    test_loader = data["test_loader"]
    evaluator = Evaluator(class_names=data["class_names"], output_dir=cfg.output_dir)

    results: dict[str, dict] = {}

    # ── Zero-shot evaluation ─────────────────────────────────────────
    logger.info(f"\n{'='*60}\n Zero-Shot CLIP Classification\n{'='*60}")
    zs_model.encode_text_prompts(data["text_prompts"], device=device)

    zs_result = evaluate_zero_shot(zs_model, test_loader, device)
    zs_metrics = evaluator.compute_metrics(zs_result["labels"], zs_result["preds"])
    results["zero_shot"] = zs_metrics

    logger.info("Zero-Shot Metrics:")
    for k, v in zs_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    evaluator.plot_confusion_matrix(
        zs_result["labels"], zs_result["preds"],
        model_name="zero_shot", top_n=20,
    )

    # ── Few-shot evaluations ─────────────────────────────────────────
    for k in few_shot_k:
        logger.info(f"\n{'='*60}\n {k}-Shot CLIP Classification (Linear Probe)\n{'='*60}")

        fs_model = CLIPFewShotClassifier(
            num_classes=data["num_classes"],
            model_name=cfg.clip_model,
            pretrained=cfg.clip_pretrained,
        )

        train_result = fs_model.train_probe(
            train_loader=data["few_shot_loaders"][k],
            test_loader=test_loader,
            epochs=cfg.probe_epochs,
            lr=cfg.probe_lr,
            weight_decay=cfg.probe_weight_decay,
            device=device,
        )

        fs_result = evaluate_few_shot(fs_model, test_loader, device)
        fs_metrics = evaluator.compute_metrics(fs_result["labels"], fs_result["preds"])
        key = f"few_shot_{k}"
        results[key] = fs_metrics

        logger.info(f"{k}-Shot Metrics:")
        for mk, mv in fs_metrics.items():
            logger.info(f"  {mk}: {mv:.4f}")

        evaluator.plot_confusion_matrix(
            fs_result["labels"], fs_result["preds"],
            model_name=key, top_n=20,
        )

    # ── Comparison ───────────────────────────────────────────────────
    evaluator.compare_models(results, task_name="Multimodal_ZeroShot_vs_FewShot")

    logger.info("\n" + "=" * 60)
    logger.info("FINAL COMPARISON — Multimodal Classification")
    logger.info("=" * 60)
    for name, m in results.items():
        logger.info(f"  {name}: " + " | ".join(f"{k}={v:.4f}" for k, v in m.items()))

    close_tb_writer()
    logger.info("Done!")


if __name__ == "__main__":
    main()
