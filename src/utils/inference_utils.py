import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from typing import List, Tuple
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("inference_utils")

def get_image_transform(image_size: int = 224) -> transforms.Compose:
    """
    Get the standard image transform for CIFAR-100 models.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

def preprocess_image(image: Image.Image, image_size: int = 224, device: str = "cpu") -> torch.Tensor:
    """
    Preprocess a PIL image for model inference.
    """
    transform = get_image_transform(image_size)
    tensor = transform(image).unsqueeze(0)
    return tensor.to(device)

def get_topk_predictions(logits: torch.Tensor, class_names: List[str], k: int = 5) -> List[Tuple[str, float]]:
    """
    Get top-K predictions from model logits.
    """
    probs = F.softmax(logits, dim=1)
    topk_probs, topk_indices = torch.topk(probs, k)
    
    topk_predictions = []
    for i in range(k):
        label = class_names[topk_indices[0][i].item()]
        score = topk_probs[0][i].item()
        topk_predictions.append((label, score))
        
    return topk_predictions

def load_model_weights(model: torch.nn.Module, checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    """
    Load model weights from a checkpoint file with robust key handling.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Robust prefix handling
    model_state_dict = model.state_dict()
    
    # helper to find the best match for a checkpoint key in the model
    def get_mapped_key(ckpt_key):
        if ckpt_key in model_state_dict:
            return ckpt_key
        # Try adding common prefixes
        for prefix in ["model.", "backbone.", "backbone.model."]:
            if f"{prefix}{ckpt_key}" in model_state_dict:
                return f"{prefix}{ckpt_key}"
        # Try removing common prefixes from checkpoint key
        for prefix in ["model.", "backbone."]:
            if ckpt_key.startswith(prefix):
                stripped = ckpt_key[len(prefix):]
                if stripped in model_state_dict:
                    return stripped
        return None

    new_state_dict = {}
    for k, v in state_dict.items():
        mapped_key = get_mapped_key(k)
        if mapped_key:
            new_state_dict[mapped_key] = v
        else:
            # Fallback for classification heads which often differ (fc vs head vs classifier)
            if "fc." in k or "classifier." in k:
                # Try to find a linear layer in our head
                for mk in model_state_dict.keys():
                    if "head" in mk and k.split(".")[-1] == mk.split(".")[-1]:
                        new_state_dict[mk] = v
                        break
            else:
                new_state_dict[k] = v

    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        logger.warning(f"Strict loading failed, trying non-strict: {e}")
        model.load_state_dict(new_state_dict, strict=False)
        
    model.to(device)
    model.eval()
    return model


# -------------------------------------------------------------------
# Enhanced inference utilities for Streamlit app
# -------------------------------------------------------------------
import json
from functools import lru_cache
import importlib
from typing import Dict, Any


class ModelLoader:
    """Unified model loading with caching."""

    def __init__(self, config_path: str = "configs/models.json"):
        with open(config_path) as f:
            self.config = json.load(f)

    @lru_cache(maxsize=5)
    def load_model(self, model_key: str, device: str = "cpu") -> torch.nn.Module:
        """Load model with architecture + weights.

        Parameters
        ----------
        model_key : str
            Model key from config (e.g., "image.cnn", "text.bilstm")
        device : str, default="cpu"
            Device to load model onto

        Returns
        -------
        torch.nn.Module
            Loaded and eval-ready model
        """
        model_info = self._get_model_info(model_key)
        model_class = self._import_class(model_info["model_class"])

        # Filter out config keys that aren't valid constructor arguments
        # If checkpoint exists, default pretrained=False unless explicitly requested
        valid_kwargs = {k: v for k, v in model_info.items()
                       if k not in ["model_class", "checkpoint_id", "class_names_source"]}
        
        if "checkpoint_id" in model_info and "pretrained" not in valid_kwargs:
            valid_kwargs["pretrained"] = False

        # Instantiate model
        model = model_class(**valid_kwargs)

        # Load weights if checkpoint available
        if "checkpoint_id" in model_info:
            from .downloader import get_model_path
            checkpoint_path = get_model_path(model_key, self.config)
            if checkpoint_path:
                model = load_model_weights(model, str(checkpoint_path), device)
            else:
                logger.warning(f"No checkpoint for {model_key}, using randomly initialized weights")
                model.to(device)
                model.eval()
        else:
            model.to(device)
            model.eval()

        return model

    def get_auxiliary_path(self, model_key: str, attr_key: str):
        """Get path to an auxiliary file (e.g., vocab_id).
        
        Parameters
        ----------
        model_key : str
            Model key from config
        attr_key : str
            Attribute key for the file ID (e.g., "vocab_id")
            
        Returns
        -------
        Optional[Path]
            Path to downloaded/cached auxiliary file
        """
        model_info = self._get_model_info(model_key)
        if attr_key not in model_info:
            return None
            
        from .downloader import download_model
        file_id = model_info[attr_key]
        cache_name = f"{model_key.replace('.', '_')}_{attr_key}_{hashlib.md5(file_id.encode()).hexdigest()[:8]}.pkl"
        
        return download_model(file_id, cache_name)

    def get_class_names(self, model_key: str) -> list:
        """Dynamically import class names from module.

        Parameters
        ----------
        model_key : str
            Model key from config

        Returns
        -------
        list
            List of class names
        """
        model_info = self._get_model_info(model_key)
        if "class_names_source" not in model_info:
            raise ValueError(f"No class_names_source defined for {model_key}")

        module_path, attr_name = model_info["class_names_source"].rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)

    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """Get model configuration info.

        Parameters
        ----------
        model_key : str
            Model key from config

        Returns
        -------
        dict
            Model configuration
        """
        return self._get_model_info(model_key).copy()

    def _get_model_info(self, model_key: str) -> dict:
        """Navigate nested config dictionary."""
        parts = model_key.split(".")
        info = self.config
        for part in parts:
            info = info.get(part, {})
            if not info:
                raise KeyError(f"Model key not found in config: {model_key}")
        return info

    def _import_class(self, class_path: str):
        """Dynamically import a class from module path."""
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


# Text preprocessing utilities
def preprocess_text_rnn(text: str, vocab, max_len: int = 256) -> torch.Tensor:
    """Preprocess text for RNN models.

    Parameters
    ----------
    text : str
        Input text
    vocab
        Vocabulary object with encode() method
    max_len : int, default=256
        Maximum sequence length

    Returns
    -------
    torch.Tensor
        Encoded token indices
    """
    # Check if vocab has encode method (from src.data.text_dataset.Vocabulary)
    if hasattr(vocab, "encode"):
        encoded = vocab.encode(text, max_len=max_len)
        return torch.tensor([encoded], dtype=torch.long)
    else:
        raise TypeError("vocab must have encode() method")


def preprocess_text_transformers(
    text: str,
    tokenizer,
    max_len: int = 256
) -> Dict[str, torch.Tensor]:
    """Preprocess text for transformer models.

    Parameters
    ----------
    text : str
        Input text
    tokenizer
        HuggingFace tokenizer
    max_len : int, default=256
        Maximum sequence length

    Returns
    -------
    dict
        Tokenized inputs with 'input_ids' and 'attention_mask'
    """
    return tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
