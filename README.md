# 🧠 Deep Learning Assignment 1 — Classification Model Comparison

> **Course**: CO5085 — Deep Learning and Applications in Computer Vision  
> **Task**: Compare classification architectures across Image, Text, and Multimodal domains

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Task 1: Image Classification](#task-1-image-classification-cnn-vs-vit)
  - [Task 2: Text Classification](#task-2-text-classification-bilstm-vs-distilbert)
  - [Task 3: Multimodal Classification](#task-3-multimodal-classification-clip-zero-shot-vs-few-shot)
- [Configuration](#configuration)
- [Outputs & Evaluation](#outputs--evaluation)
- [Running on Google Colab](#running-on-google-colab)
- [Maintenance & Development Guide](#maintenance--development-guide)

---

## Overview

This project implements and compares deep learning classification models across three data modalities:

| Task | Dataset | Model A | Model B | Classes |
|---|---|---|---|---|
| **Image** | CIFAR-100 (60K images) | ResNet-50 (CNN) | ViT-Small (Vision Transformer) | 100 |
| **Text** | DBpedia-14 (560K articles) | BiLSTM + Self-Attention | DistilBERT Fine-tuning | 14 |
| **Multimodal** | CIFAR-100 + CLIP prompts | CLIP Zero-shot | CLIP Few-shot (Linear Probe) | 100 |

### Bonus Features

- **Interpretability**: Grad-CAM (CNN), attention map visualization (ViT, BiLSTM, DistilBERT)
- **Advanced Augmentation**: MixUp, CutMix, RandAugment
- **Deployment**: Streamlit web demo with all 3 tasks

---

## Project Structure

```
DeepLearningAsm1_Classification/
│
├── configs/                        # YAML hyperparameter configs
│   ├── image.yaml                  # Task 1 config
│   ├── text.yaml                   # Task 2 config
│   └── multimodal.yaml             # Task 3 config
│
├── src/                            # Source code (importable package)
│   ├── data/                       # Data loading & preprocessing
│   │   ├── image_dataset.py        # CIFAR-100 with RandAugment
│   │   ├── text_dataset.py         # DBpedia-14 with vocab builder + tokenizer
│   │   └── multimodal_dataset.py   # CIFAR-100 for CLIP (zero/few-shot)
│   │
│   ├── models/                     # Model architectures
│   │   ├── cnn.py                  # ResNet-50/34/50 classifier
│   │   ├── vit.py                  # ViT-Small via timm
│   │   ├── rnn.py                  # BiLSTM + Self-Attention
│   │   ├── transformer_text.py     # DistilBERT fine-tuning
│   │   ├── clip_zeroshot.py        # CLIP zero-shot classifier
│   │   └── clip_fewshot.py         # CLIP linear probe (few-shot)
│   │
│   ├── engine/                     # Training & evaluation engine
│   │   ├── trainer.py              # Generic Trainer (AMP, early stopping, checkpointing)
│   │   └── evaluator.py            # Metrics, confusion matrices, comparison charts
│   │
│   ├── interpret/                  # Interpretability tools
│   │   ├── gradcam.py              # Grad-CAM for CNNs
│   │   └── attention_vis.py        # ViT & text attention visualization
│   │
│   └── utils/                      # Shared utilities
│       ├── config.py               # YAML config loader with CLI overrides
│       ├── seed.py                 # Reproducibility (torch + numpy + random)
│       ├── logger.py               # Console + file + TensorBoard logging
│       └── augmentations.py        # MixUp & CutMix implementations
│
├── scripts/                        # Entry-point scripts
│   ├── train_image.py              # Train & evaluate CNN vs ViT
│   ├── train_text.py               # Train & evaluate BiLSTM vs DistilBERT
│   ├── train_multimodal.py         # Evaluate CLIP zero-shot vs few-shot
│   └── streamlit_app.py            # Streamlit web demo (3 tabs)
│
├── notebooks/                      # Self-contained Colab notebooks (if present)
├── outputs/                        # Generated outputs (git-ignored)
│   ├── checkpoints/                # Saved model weights (.pt)
│   ├── logs/                       # TensorBoard logs
│   └── reports/                    # Confusion matrices, comparison charts
│
├── pyproject.toml                  # Project dependencies
├── Report.ipynb                    # Report notebook
└── README.md                       # This file
```

---

## Requirements

- **Python** ≥ 3.10
- **GPU** recommended (NVIDIA with CUDA) — works on CPU but very slow
- **Disk** ~2 GB for datasets (auto-downloaded on first run)
- **RAM** ≥ 8 GB system, ≥ 12 GB GPU VRAM (for T4 or better)

### Key Dependencies

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Deep learning framework |
| `timm` | Pre-trained ViT models |
| `transformers`, `datasets` | HuggingFace (DistilBERT, DBpedia-14) |
| `open-clip-torch` | CLIP models for multimodal task |
| `scikit-learn` | Metrics (accuracy, F1, confusion matrix) |
| `gradio` | Web demo app |
| `matplotlib`, `seaborn` | Visualization |
| `tensorboard` | Training log visualization |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/khasang12/DeepLearningAsm1_Classification.git
cd DeepLearningAsm1_Classification

# 2. Install dependencies via Poetry (recommended)
# Activated environment will be created automatically
poetry install

# 3. Activate the environment (Poetry 2.0+)
poetry env activate

# 4. Verify installation
poetry run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

> [!NOTE]
> If you don't have Poetry installed, follow the [official installation guide](https://python-poetry.org/docs/#installation).

---

## Usage

All training scripts should be run through Poetry to ensure dependencies are available:
```bash
poetry run python scripts/train_<task>.py --config configs/<task>.yaml [--override_key value]
```

### Task 1: Image Classification (CNN vs ViT)

```bash
# Full training (50 epochs, default config)
python scripts/train_image.py --config configs/image.yaml

# Quick test (2 epochs, small batch)
python scripts/train_image.py --config configs/image.yaml --epochs 2 --batch_size 32

# With MixUp augmentation
python scripts/train_image.py --config configs/image.yaml --augmentation mixup

# With CutMix augmentation
python scripts/train_image.py --config configs/image.yaml --augmentation cutmix
```

**What it does:**
1. Downloads CIFAR-100 (auto, first run only)
2. Trains ResNet-50 with early stopping → saves `outputs/image/checkpoints/cnn_best.pt`
3. Trains ViT-Small with early stopping → saves `outputs/image/checkpoints/vit_best.pt`
4. Evaluates both on the test set
5. Generates confusion matrices, training curves, and comparison charts
6. Generates Grad-CAM visualizations for the CNN

---

### Task 2: Text Classification (BiLSTM vs DistilBERT)

```bash
# Full training
python scripts/train_text.py --config configs/text.yaml

# Quick test
python scripts/train_text.py --config configs/text.yaml --epochs 2 --batch_size 32
```

**What it does:**
1. Downloads DBpedia-14 from HuggingFace (560K articles, 14 classes)
2. Builds vocabulary for RNN, tokenizes with DistilBERT tokenizer
3. Trains BiLSTM (self-attention pooling) → saves checkpoint
4. Trains DistilBERT (fine-tuning) → saves checkpoint
5. Generates per-class classification reports and comparison charts

---

### Task 3: Multimodal Classification (CLIP Zero-shot vs Few-shot)

```bash
# Full evaluation (K=1,5,10,20 shots)
python scripts/train_multimodal.py --config configs/multimodal.yaml

# Quick test (1-shot only)
python scripts/train_multimodal.py --config configs/multimodal.yaml --few_shot_k 1
```

**What it does:**
1. Loads CLIP ViT-B/32 (pre-trained, frozen)
2. Runs zero-shot classification using text prompts
3. Trains linear probes for K-shot settings (1, 5, 10, 20)
4. Compares accuracy across all settings

---

### Streamlit Demo App

The refined Streamlit application provides a better interactive experience with model caching and visualization support.

```bash
# Run via Poetry
poetry run streamlit run scripts/streamlit_app.py
```

Three interactive tabs:
- **🖼 Image** — Upload image → classify with CNN & ViT + Grad-CAM/Attention visualization.
- **📝 Text** — (Template) Sentiment analysis workflow.
- **🔗 Multimodal** — (Template) CLIP zero-shot classification workflow.

---

## Configuration

All hyperparameters are defined in YAML files under `configs/`. Any parameter can be overridden via CLI:

```bash
# Override a top-level parameter
python scripts/train_image.py --lr 0.0005 --epochs 100

# Override a nested parameter (dot notation)
python scripts/train_image.py --models.cnn.pretrained false

# Override list parameters
python scripts/train_multimodal.py --few_shot_k "[1,5]"
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `seed` | 42 | Random seed for reproducibility |
| `batch_size` | 64–128 | Samples per batch (adjust for GPU memory) |
| `epochs` | 10–50 | Maximum training epochs |
| `lr` | 0.001–2e-5 | Learning rate (task-dependent) |
| `early_stopping_patience` | 3–7 | Epochs without improvement before stopping |
| `augmentation` | null | `"mixup"` or `"cutmix"` (image task only) |
| `num_workers` | 4 | DataLoader parallel workers |

---

## Outputs & Evaluation

After training, all outputs are saved to `outputs/<task>/`:

```
outputs/
├── image/
│   ├── checkpoints/          # cnn_best.pt, vit_best.pt
│   ├── logs/                 # TensorBoard logs
│   ├── reports/              # confusion_matrix_cnn.png, comparison_Image_CNN_vs_ViT.png
│   └── gradcam/              # Grad-CAM visualizations
├── text/
│   ├── checkpoints/          # rnn_best.pt, transformer_best.pt
│   ├── logs/
│   └── reports/              # confusion matrices, comparison charts
└── multimodal/
    ├── logs/
    └── reports/              # zero-shot vs few-shot comparison
```

### Viewing TensorBoard Logs

```bash
tensorboard --logdir outputs/
# Open http://localhost:6006
```

### Evaluation Metrics

All tasks report:
- **Accuracy** — Overall correct predictions
- **Precision** (macro) — Per-class precision averaged
- **Recall** (macro) — Per-class recall averaged
- **F1-score** (macro) — Harmonic mean of precision and recall
- **Confusion Matrix** — Heatmap visualization (saved as PNG)
- **Classification Report** — Full per-class breakdown (printed to console)

---

## Running on Google Colab

1. **Set GPU runtime**: `Runtime → Change runtime type → GPU (T4)`

2. **Setup**:
```python
!git clone https://github.com/khasang12/DeepLearningAsm1_Classification.git
%cd DeepLearningAsm1_Classification
!pip install -e . -q
```

3. **Train** (optimized for T4 GPU):
```python
# Task 1
!python scripts/train_image.py --batch_size 256 --num_workers 4

# Task 2
!python scripts/train_text.py --batch_size 64 --num_workers 4

# Task 3
!python scripts/train_multimodal.py --batch_size 512 --num_workers 4
```

4. **Download results**:
```python
import shutil
shutil.make_archive("outputs", "zip", ".", "outputs")
from google.colab import files
files.download("outputs.zip")
```

### Recommended Batch Sizes by GPU

| GPU | Image | Text | Multimodal |
|---|---|---|---|
| T4 (16 GB) | 256 | 64 | 512 |
| A100 (40 GB) | 512 | 128 | 1024 |
| CPU (no GPU) | 32 | 16 | 64 |

---

## Maintenance & Development Guide

### Adding a New Model

1. Create a new file in `src/models/`, e.g. `src/models/my_model.py`
2. Implement a `nn.Module` subclass with a `forward(x) → logits` signature
3. Register it in `src/models/__init__.py`
4. Add configuration in the appropriate `configs/*.yaml`
5. Update the training script to instantiate and train your model

### Adding a New Dataset

1. Create a new file in `src/data/`, e.g. `src/data/my_dataset.py`
2. Implement a `get_<name>_dataloaders()` function returning `train_loader`, `val_loader`, `test_loader`, `class_names`, `num_classes`
3. Register it in `src/data/__init__.py`
4. Update the training script to use the new data source

### Code Style & Quality

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
black src/ scripts/

# Lint
ruff check src/ scripts/

# Run tests (if added)
pytest tests/
```

### Common Issues & Troubleshooting

| Issue | Solution |
|---|---|
| `CUDA out of memory` | Reduce `batch_size` by half |
| `num_workers` causing crashes | Set `num_workers: 0` on Windows or low-RAM systems |
| Slow training on CPU | Use Google Colab with GPU runtime |
| Dataset download fails | Check internet connection; datasets are cached after first download in `./data/` or `~/.cache/huggingface/` |
| `ModuleNotFoundError: src` | Run `poetry install` or `pip install -e .` |

### File Responsibilities

| Layer | Files | Role |
|---|---|---|
| **Entry points** | `scripts/train_*.py`, `scripts/demo_app.py` | CLI commands, orchestration |
| **Models** | `src/models/*.py` | Architecture definitions only |
| **Data** | `src/data/*.py` | Dataset loading, transforms, DataLoaders |
| **Engine** | `src/engine/trainer.py`, `evaluator.py` | Training loop, metrics, visualization |
| **Utilities** | `src/utils/*.py` | Config, logging, seeding, augmentation |
| **Interpret** | `src/interpret/*.py` | Grad-CAM, attention maps |
| **Config** | `configs/*.yaml` | All hyperparameters (no hardcoded values) |

---

## License

This project is for educational purposes as part of the CO5085 course.
