"""Gradio demo app for all three classification tasks."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import gradio as gr
from PIL import Image


# ──────────────────────────────────────────────────────────────────────
# Lazy model loading (loaded on first request)
# ──────────────────────────────────────────────────────────────────────

_cache: dict = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_image_models():
    """Load ResNet and ViT models for image classification."""
    if "image" not in _cache:
        from src.data.image_dataset import CIFAR100_CLASSES
        from src.models.cnn import ResNetClassifier
        from src.models.vit import ViTClassifier
        import torchvision.transforms as T

        cnn = ResNetClassifier(num_classes=100, pretrained=True).to(DEVICE).eval()
        vit = ViTClassifier(num_classes=100, pretrained=True).to(DEVICE).eval()

        # Try loading trained checkpoints
        cnn_ckpt = Path("outputs/image/checkpoints/cnn_best.pt")
        vit_ckpt = Path("outputs/image/checkpoints/vit_best.pt")
        if cnn_ckpt.exists():
            cnn.load_state_dict(torch.load(cnn_ckpt, map_location=DEVICE)["model_state_dict"])
        if vit_ckpt.exists():
            vit.load_state_dict(torch.load(vit_ckpt, map_location=DEVICE)["model_state_dict"])

        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ])

        _cache["image"] = {
            "cnn": cnn, "vit": vit, "transform": transform,
            "classes": CIFAR100_CLASSES,
        }
    return _cache["image"]


def _get_text_models():
    """Load BiLSTM and DistilBERT for text classification."""
    if "text" not in _cache:
        from src.data.text_dataset import DBPEDIA_CLASSES, Vocabulary
        from src.models.rnn import BiLSTMClassifier
        from src.models.transformer_text import DistilBERTClassifier
        from transformers import AutoTokenizer

        # Build a minimal vocab (use saved if available)
        vocab = Vocabulary(max_size=50_000)
        vocab.build([])  # Empty vocab — will use UNK for unseen words

        rnn = BiLSTMClassifier(
            vocab_size=len(vocab), num_classes=4, embed_dim=300,
            hidden_dim=256, num_layers=2, dropout=0.3,
        ).to(DEVICE).eval()

        transformer = DistilBERTClassifier(
            num_classes=14, pretrained=True,
        ).to(DEVICE).eval()

        rnn_ckpt = Path("outputs/text/checkpoints/rnn_best.pt")
        tf_ckpt = Path("outputs/text/checkpoints/transformer_best.pt")
        if rnn_ckpt.exists():
            rnn.load_state_dict(torch.load(rnn_ckpt, map_location=DEVICE))
        if tf_ckpt.exists():
            transformer.load_state_dict(torch.load(tf_ckpt, map_location=DEVICE))

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        _cache["text"] = {
            "rnn": rnn, "transformer": transformer,
            "vocab": vocab, "tokenizer": tokenizer,
            "classes": DBPEDIA_CLASSES,
        }
    return _cache["text"]


def _get_clip_model():
    """Load CLIP for zero-shot classification."""
    if "clip" not in _cache:
        from src.models.clip_zeroshot import CLIPZeroShotClassifier
        from src.data.image_dataset import CIFAR100_CLASSES

        model = CLIPZeroShotClassifier(model_name="ViT-B-32", pretrained="openai")
        model.to(DEVICE)
        prompts = [f"a photo of a {c}" for c in CIFAR100_CLASSES]
        model.encode_text_prompts(prompts, device=DEVICE)

        _cache["clip"] = {"model": model, "classes": CIFAR100_CLASSES}
    return _cache["clip"]


# ──────────────────────────────────────────────────────────────────────
# Prediction functions
# ──────────────────────────────────────────────────────────────────────

def classify_image(image: Image.Image) -> dict[str, dict[str, float]]:
    """Classify an image using CNN and ViT."""
    models = _get_image_models()
    img_tensor = models["transform"](image).unsqueeze(0).to(DEVICE)

    results = {}
    for name in ["cnn", "vit"]:
        with torch.no_grad():
            logits = models[name](img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top5_probs, top5_idx = probs.topk(5)
        results[name.upper()] = {
            models["classes"][i]: float(p)
            for p, i in zip(top5_probs, top5_idx)
        }

    return results.get("CNN", {}), results.get("VIT", {})


def classify_text(text: str) -> dict[str, dict[str, float]]:
    """Classify text using BiLSTM and DistilBERT."""
    models = _get_text_models()
    classes = models["classes"]

    # RNN
    encoded = models["vocab"].encode(text, max_len=256)
    rnn_input = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        rnn_logits = models["rnn"](rnn_input)
    rnn_probs = torch.softmax(rnn_logits, dim=1)[0]

    # Transformer
    tf_enc = models["tokenizer"](
        text, padding="max_length", truncation=True,
        max_length=256, return_tensors="pt",
    )
    input_ids = tf_enc["input_ids"].to(DEVICE)
    attention_mask = tf_enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        tf_logits = models["transformer"](input_ids, attention_mask=attention_mask)
    tf_probs = torch.softmax(tf_logits, dim=1)[0]

    rnn_result = {classes[i]: float(rnn_probs[i]) for i in range(len(classes))}
    tf_result = {classes[i]: float(tf_probs[i]) for i in range(len(classes))}
    return rnn_result, tf_result


def classify_zero_shot(image: Image.Image) -> dict[str, float]:
    """Classify an image using CLIP zero-shot."""
    clip = _get_clip_model()
    preprocess = clip["model"].get_preprocess()
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, probs = clip["model"].predict(img_tensor)
    probs = probs[0]
    top10_probs, top10_idx = probs.topk(10)

    return {clip["classes"][i]: float(p) for p, i in zip(top10_probs, top10_idx)}


# ──────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    """Build the Gradio demo application."""
    with gr.Blocks(
        title="DL Assignment 1 — Classification Demo",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# 🧠 Deep Learning Classification Demo\n"
            "Compare different model architectures across image, text, and multimodal tasks."
        )

        with gr.Tab("🖼 Image Classification"):
            gr.Markdown("**CNN (ResNet-18) vs Vision Transformer (ViT)**")
            with gr.Row():
                img_input = gr.Image(type="pil", label="Upload Image")
                with gr.Column():
                    cnn_output = gr.Label(num_top_classes=5, label="CNN (ResNet-18)")
                    vit_output = gr.Label(num_top_classes=5, label="ViT")
            img_btn = gr.Button("Classify", variant="primary")
            img_btn.click(classify_image, inputs=img_input, outputs=[cnn_output, vit_output])

        with gr.Tab("📝 Text Classification"):
            gr.Markdown("**BiLSTM vs DistilBERT** — DBpedia-14 (14 categories)")
            text_input = gr.Textbox(
                lines=4, placeholder="Enter a news article...",
                label="Input Text",
            )
            with gr.Row():
                rnn_output = gr.Label(label="BiLSTM")
                tf_output = gr.Label(label="DistilBERT")
            text_btn = gr.Button("Classify", variant="primary")
            text_btn.click(classify_text, inputs=text_input, outputs=[rnn_output, tf_output])

        with gr.Tab("🔗 Multimodal (CLIP Zero-Shot)"):
            gr.Markdown("**CLIP ViT-B/32 Zero-Shot** — Classify any image into 100 categories")
            clip_input = gr.Image(type="pil", label="Upload Image")
            clip_output = gr.Label(num_top_classes=10, label="CLIP Predictions (Top-10)")
            clip_btn = gr.Button("Classify", variant="primary")
            clip_btn.click(classify_zero_shot, inputs=clip_input, outputs=clip_output)

        gr.Markdown("---\n*CO5085 — Deep Learning & Applications in Computer Vision*")

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
