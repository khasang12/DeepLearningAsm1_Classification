# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep learning assignment comparing classification architectures across three modalities:
1. **Image Classification** (CIFAR-100): ResNet-50 (CNN) vs ViT-Small (Vision Transformer) vs MobileNetV3-Small
2. **Text Classification** (DBpedia-14): BiLSTM + Self-Attention vs DistilBERT fine-tuning
3. **Multimodal Classification** (CIFAR-100 + CLIP prompts): CLIP Zero-shot vs CLIP Few-shot (Linear Probe)

The project includes interpretability features (Grad-CAM, attention visualization), advanced augmentation (MixUp, CutMix), and a Streamlit web demo.

## Development Commands

### Installation & Setup
```bash
# Install dependencies via Poetry (Recommended)
poetry install

# Install via Pip (Optional)
pip install -e .
```

### Training Scripts
All training scripts follow the same pattern with YAML configs and CLI overrides:
```bash
# Image classification (CNN vs ViT)
python scripts/train_image.py --config configs/image.yaml [--override_key value]

# Text classification (BiLSTM vs DistilBERT)
python scripts/train_text.py --config configs/text.yaml [--override_key value]

# Multimodal classification (CLIP zero-shot vs few-shot)
python scripts/train_multimodal.py --config configs/multimodal.yaml [--override_key value]
```

### Common CLI Overrides
```bash
# Override top-level parameters
--lr 0.0005 --epochs 100 --batch_size 64

# Override nested parameters (dot notation)
--models.cnn.pretrained false --models.vit.name vit_base_patch16_224

# List parameters
--few_shot_k "[1,5,10]"

# Quick testing
python scripts/train_image.py --epochs 2 --batch_size 32
```

### Code Quality
```bash
# Format code
black src/ scripts/

# Lint
ruff check src/ scripts/

# Run tests (when implemented)
pytest tests/
```

### Demo App
```bash
# Launch Streamlit web interface (Recommended)
poetry run streamlit run scripts/streamlit_app.py

# Launch Gradio web interface (Legacy)
python scripts/demo_app.py
```

### TensorBoard
```bash
# View training logs
tensorboard --logdir outputs/
# Access at http://localhost:6006
```

## Architecture & Structure

### Configuration System
- **Location**: `configs/*.yaml` (image.yaml, text.yaml, multimodal.yaml)
- **Loading**: `src.utils.config.load_config()` parses YAML and merges CLI overrides
- **Access**: Configuration is returned as a nested `SimpleNamespace` object
- **Override Syntax**: Dotted notation for nested keys (e.g., `--models.cnn.pretrained false`)

### Source Code Organization (`src/`)
- **`src/data/`**: Dataset loading and preprocessing
  - `image_dataset.py`: CIFAR-100 with RandAugment
  - `text_dataset.py`: DBpedia-14 with vocab builder + tokenizer
  - `multimodal_dataset.py`: CIFAR-100 for CLIP (zero/few-shot)
- **`src/models/`**: Model architectures
  - `cnn.py`: ResNet-50/34/50 classifier
  - `vit.py`: ViT-Small via timm
  - `rnn.py`: BiLSTM + Self-Attention
  - `transformer_text.py`: DistilBERT fine-tuning
  - `clip_zeroshot.py`: CLIP zero-shot classifier
  - `clip_fewshot.py`: CLIP linear probe (few-shot)
- **`src/engine/`**: Training & evaluation engine
  - `trainer.py`: Generic Trainer with AMP, early stopping, checkpointing
  - `evaluator.py`: Metrics, confusion matrices, comparison charts
- **`src/interpret/`**: Interpretability tools
  - `gradcam.py`: Grad-CAM for CNNs
  - `attention_vis.py`: ViT & text attention visualization
- **`src/utils/`**: Shared utilities
  - `config.py`: YAML config loader with CLI overrides
  - `seed.py`: Reproducibility (torch + numpy + random)
  - `logger.py`: Console + file + TensorBoard logging
  - `augmentations.py`: MixUp & CutMix implementations

### Training Flow
1. **Entry Point**: Scripts in `scripts/` load config, set up logging, and orchestrate training
2. **Data Loading**: Call `get_*_dataloaders()` from appropriate data module
3. **Model Instantiation**: Models are created via factory functions in training scripts
4. **Training**: `Trainer.fit()` handles the training loop with validation and early stopping
5. **Evaluation**: `Evaluator` computes metrics, generates plots, and saves reports
6. **Outputs**: Saved to `outputs/<task>/` (checkpoints, logs, reports, visualizations)

### Key Design Patterns
- **Separation of Concerns**: Models define architecture only; training logic is in `Trainer`
- **Configuration-Driven**: All hyperparameters in YAML files, no hardcoded values
- **Modular Data Loading**: Each dataset provides a `get_*_dataloaders()` function
- **Reusable Trainer**: Single `Trainer` class handles all training scenarios with optional augmentation
- **Unified Evaluation**: `Evaluator` provides consistent metrics and visualization across tasks

### Output Structure
```
outputs/
├── image/
│   ├── checkpoints/          # cnn_best.pt, vit_best.pt
│   ├── logs/                 # TensorBoard logs
│   ├── reports/              # confusion_matrix_cnn.png, comparison charts
│   └── gradcam/              # Grad-CAM visualizations
├── text/
│   ├── checkpoints/          # rnn_best.pt, transformer_best.pt
│   ├── logs/
│   └── reports/
└── multimodal/
    ├── logs/
    └── reports/
```

## Adding New Components

### Adding a New Model
1. Create file in `src/models/` (e.g., `my_model.py`)
2. Implement `nn.Module` subclass with `forward(x) → logits` signature
3. Register in `src/models/__init__.py`
4. Add configuration in appropriate `configs/*.yaml`
5. Update training script to instantiate and train the model

### Adding a New Dataset
1. Create file in `src/data/` (e.g., `my_dataset.py`)
2. Implement `get_<name>_dataloaders()` returning `train_loader`, `val_loader`, `test_loader`, `class_names`, `num_classes`
3. Register in `src/data/__init__.py`
4. Update training script to use the new data source

## Dependencies (Poetry)
Key dependencies (see `pyproject.toml` or `poetry.lock` for complete list):
- `torch`, `torchvision`: Deep learning framework
- `timm`: Pre-trained ViT models
- `transformers`, `datasets`: HuggingFace (DistilBERT, DBpedia-14)
- `open-clip-torch`: CLIP models for multimodal task
- `streamlit`: Web demo app
- `scipy`: Technical computing (used for attention upsampling)
- Development: `pytest`, `ruff`, `black`

## Important Notes
- **GPU Requirements**: Training requires GPU (NVIDIA with CUDA). Adjust `batch_size` based on available VRAM.
- **Reproducibility**: Set `seed` in config for deterministic results.
- **Dataset Caching**: Datasets are auto-downloaded on first run and cached locally.
- **Output Directory**: All outputs go to `outputs/` (git-ignored).
- **Demo App**: The Gradio demo (`scripts/demo_app.py`) provides interactive testing of all three tasks.