"""Streamlit inference app for DL Assignment 1 models."""

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import sys
import re
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.inference_utils import (
    ModelLoader, preprocess_image, get_topk_predictions,
)
from src.utils.downloader import get_model_path, clear_cache
from src.utils.logger import get_logger

import pickle
import matplotlib.pyplot as plt

logger = get_logger("streamlit_app")

# Page configuration
st.set_page_config(
    page_title="DL Assignment 1 - Model Inference",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def get_model_loader():
    """Cache the model loader across Streamlit sessions."""
    try:
        return ModelLoader("configs/models.json")
    except FileNotFoundError:
        st.error("Configuration file not found: configs/models.json")
        st.stop()

def setup_sidebar():
    """Create sidebar configuration options."""
    st.sidebar.header("Configuration")

    # Device selection
    use_gpu = st.sidebar.checkbox("Use GPU if available", True)
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    st.sidebar.info(f"Using device: **{device}**")

    # Model management
    st.sidebar.subheader("Model Management")
    if st.sidebar.button("Clear Model Cache"):
        count = clear_cache()
        st.sidebar.success(f"Cleared {count} cached model files")

    # Force download option
    force_download = st.sidebar.checkbox("Force re-download models", False)

    return device, force_download


# =====================================================================
# Tab 1: Image Classification
# =====================================================================

def image_classification_tab(model_loader, device):
    """Image classification tab."""
    st.header("🖼 Image Classification (CIFAR-100)")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Image upload
        image_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
            key="image_upload"
        )

        # Model selection
        model_options = ["image.cnn", "image.vit"]
        # Check if mobilenetv3 exists in config
        try:
            model_loader.get_model_info("image.mobilenetv3")
            model_options.append("image.mobilenetv3")
        except KeyError:
            pass

        model_type = st.selectbox(
            "Select Model",
            model_options,
            help="CNN: ResNet-50, ViT: Vision Transformer"
        )

        # Additional options
        show_gradcam = st.checkbox("Show Grad-CAM (CNN)/ Attention (ViT) visualization", value=True)

    with col2:
        if image_file:
            image = Image.open(image_file).convert("RGB")
            st.image(image, caption="Uploaded Image")
            st.caption(f"Raw Image Size: {image.size[0]}x{image.size[1]} (W x H)")

    if image_file and model_type and st.button("Run Inference", type="primary", key="image_infer"):
        with st.spinner("Loading model and running inference..."):
            try:
                # Load model
                model = model_loader.load_model(model_type, device=device)
                class_names = model_loader.get_class_names(model_type)

                # Preprocess and predict
                tensor = preprocess_image(image, device=device)
                st.info(f"Processed Tensor Shape for Inference: {list(tensor.shape)} (B x C x H x W)")
                with torch.no_grad():
                    logits = model(tensor)

                predictions = get_topk_predictions(logits, class_names, k=5)

                # Display results
                st.subheader("Top-5 Predictions")
                for label, score in predictions:
                    st.progress(float(score), text=f"{label}: {score:.3f}")

                # Show raw scores
                with st.expander("See detailed scores"):
                    for label, score in predictions:
                        st.write(f"**{label}**: {score:.4f}")

                # Grad-CAM for CNN
                if show_gradcam and model_type == "image.cnn":
                    try:
                        from src.interpret.gradcam import GradCAM
                        grad_cam = GradCAM(model, model.get_target_layer())
                        heatmap = grad_cam.generate(tensor)

                        # Create visualization
                        fig = grad_cam.visualize(
                            tensor[0].cpu(),
                            heatmap[0], # generate returns numpy (B, H, W)
                            title=f"Grad-CAM: {predictions[0][0]}"
                        )
                        st.pyplot(fig)
                        plt.close(fig) # Avoid memory leak

                        # Cleanup hooks
                        grad_cam.remove_hooks()
                    except Exception as e:
                        st.warning(f"Could not generate Grad-CAM: {e}")

                # Attention visualization for ViT
                if model_type == "image.vit":
                    try:
                        from src.interpret.attention_vis import visualize_vit_attention

                        # We need to run inference again with hooks registered IF they weren't
                        if hasattr(model, "register_attention_hooks"):
                            model.register_attention_hooks()
                            # Re-run forward to capture attention
                            with torch.no_grad():
                                _ = model(tensor)

                            attn_maps = model.get_attention_maps()
                            if attn_maps:
                                fig = visualize_vit_attention(
                                    attn_maps,
                                    image=tensor[0],
                                    layer=-1 # Show last layer
                                )
                                st.subheader("Transformer Attention Map")
                                st.pyplot(fig)
                                plt.close(fig)

                            model.remove_attention_hooks()
                    except Exception as e:
                        st.warning(f"Could not generate attention map: {e}")

            except Exception as e:
                st.error(f"Inference failed: {e}")
                logger.exception("Image inference error")


# =====================================================================
# Tab 2: Text Classification
# =====================================================================

def preprocess_text(text: str) -> str:
    """Clean text matching text_pretrained.py pipeline."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def text_classification_tab(model_loader, device):
    """Text classification tab."""
    st.header("📝 Text Classification (DBpedia-14)")

    st.markdown("""
    Classify text into **14 categories**: Company, Educational Institution, Artist,
    Athlete, Office Holder, Transportation, Building, Natural Place, Village,
    Animal, Plant, Album, Film, Written Work.
    """)

    text_input = st.text_area(
        "Enter text for classification:",
        height=150,
        placeholder="Enter a Wikipedia article or description...\n\nExample: 'Eucalyptus is a genus of over seven hundred species of flowering plants in the family Myrtaceae.'",
    )

    model_type = st.selectbox(
        "Select Model",
        ["text.bert", "text.bilstm"],
        format_func=lambda x: {
            "text.bert": "BERT (Frozen Backbone — Linear Probing)",
            "text.bilstm": "BiLSTM (Frozen GloVe Embeddings)",
        }.get(x, x),
        key="text_model"
    )

    if text_input and model_type and st.button("Classify Text", type="primary", key="text_infer"):
        with st.spinner("Processing text..."):
            try:
                class_names = model_loader.get_class_names(model_type)
                cleaned = preprocess_text(text_input)
                st.info(f"Cleaned text: *{cleaned[:100]}...*" if len(cleaned) > 100 else f"Cleaned text: *{cleaned}*")

                if model_type == "text.bert":
                    # ----- BERT path -----
                    from transformers import BertTokenizer
                    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

                    encoding = tokenizer(
                        cleaned,
                        padding="max_length",
                        truncation=True,
                        max_length=256,
                        return_tensors="pt"
                    )

                    input_ids = encoding["input_ids"].to(device)
                    attention_mask = encoding["attention_mask"].to(device)

                    # Load model
                    model = model_loader.load_model(model_type, device=device)

                    with torch.no_grad():
                        logits = model(input_ids=input_ids, attention_mask=attention_mask)

                    predictions = get_topk_predictions(logits, class_names, k=5)

                elif model_type == "text.bilstm":
                    # ----- BiLSTM path -----
                    # Load vocab from cache/drive
                    vocab_path = model_loader.get_auxiliary_path(model_type, "vocab_id")
                    if not vocab_path:
                        st.error("No vocabulary file defined for BiLSTM. Please set `vocab_id` in models.json.")
                        return

                    with open(vocab_path, "rb") as f:
                        vocab = pickle.load(f)

                    # Tokenize: split → lookup → pad/truncate
                    max_len = 256
                    words = cleaned.split()
                    if isinstance(vocab, dict):
                        unk_idx = vocab.get("<UNK>", vocab.get("<unk>", 1))
                        token_ids = [vocab.get(w, unk_idx) for w in words]
                    else:
                        token_ids = vocab.encode(cleaned, max_len=max_len)

                    # Pad / truncate to max_len
                    if len(token_ids) > max_len:
                        token_ids = token_ids[:max_len]
                    else:
                        token_ids = token_ids + [0] * (max_len - len(token_ids))

                    input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

                    # Build model with ACTUAL vocab size (not config's hardcoded value)
                    from src.models.rnn import BiLSTMClassifier
                    from src.utils.inference_utils import load_model_weights
                    actual_vocab_size = len(vocab) if isinstance(vocab, dict) else 50000
                    model = BiLSTMClassifier(
                        num_classes=14,
                        vocab_size=actual_vocab_size,
                        embed_dim=100,
                        hidden_dim=128,
                        num_layers=2,
                        dropout=0.3,
                    )

                    # Load trained weights
                    from src.utils.downloader import get_model_path
                    checkpoint_path = get_model_path(model_type, model_loader.config)
                    if checkpoint_path:
                        model = load_model_weights(model, str(checkpoint_path), device)
                    else:
                        st.warning("No checkpoint found — using untrained model")
                        model.to(device)
                        model.eval()

                    with torch.no_grad():
                        logits = model(input_ids=input_tensor)

                    predictions = get_topk_predictions(logits, class_names, k=5)

                # Display results
                st.subheader("Top-5 Predictions")
                for label, score in predictions:
                    st.progress(float(score), text=f"{label}: {score:.3f}")

                with st.expander("See detailed scores"):
                    for label, score in predictions:
                        st.write(f"**{label}**: {score:.4f}")

            except Exception as e:
                st.error(f"Text classification failed: {e}")
                logger.exception("Text inference error")


# =====================================================================
# Tab 3: Multimodal Classification
# =====================================================================

def multimodal_classification_tab(model_loader, device):
    """Multimodal classification tab (CLIP zero-shot and few-shot)."""
    st.header("🔗 Multimodal Classification (CLIP — COCO)")

    st.markdown("""
    Classify images into **COCO supercategories**: animal, food, furniture, indoor,
    kitchen, outdoor, person, sports, vehicle.
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        clip_image = st.file_uploader(
            "Upload image for CLIP classification",
            type=["jpg", "jpeg", "png"],
            key="clip_upload"
        )

        model_type = st.selectbox(
            "Select Approach",
            ["multimodal.clip_zero", "multimodal.clip_few"],
            format_func=lambda x: {
                "multimodal.clip_zero": "Zero-Shot (text prompts, no training data)",
                "multimodal.clip_few": "Few-Shot Prototype (K=16 support examples)",
            }.get(x, x),
            key="clip_model"
        )

        # Optional text caption for few-shot
        text_caption = ""
        if model_type == "multimodal.clip_few":
            text_caption = st.text_input(
                "Optional: enter a text caption for multimodal features",
                placeholder="e.g., 'a dog sitting on a couch'"
            )

    with col2:
        if clip_image:
            image = Image.open(clip_image).convert("RGB")
            st.image(image, caption="Uploaded Image")
            st.caption(f"Raw Image Size: {image.size[0]}x{image.size[1]} (W x H)")

    if clip_image and model_type and st.button("Run CLIP Inference", type="primary", key="clip_infer"):
        with st.spinner("Running CLIP inference..."):
            try:
                class_names = model_loader.get_class_names(model_type)

                if model_type == "multimodal.clip_zero":
                    # ----- Zero-Shot path -----
                    model = model_loader.load_model(model_type, device=device)
                    processor = model.get_processor()

                    # Encode text prompts for all classes
                    prompts = [f"a photo of a {c}" for c in class_names]
                    model.encode_text_prompts(prompts, device=device)

                    # Process image
                    inputs = processor(images=image, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(device)
                    st.info(f"Processed Tensor Shape: {list(pixel_values.shape)} (B x C x H x W)")

                    logits = model(pixel_values)
                    predictions = get_topk_predictions(logits, class_names, k=len(class_names))

                elif model_type == "multimodal.clip_few":
                    # ----- Few-Shot Prototype path -----
                    model = model_loader.load_model(model_type, device=device)
                    processor = model.get_processor()

                    # If prototypes were loaded, try to get class_names from checkpoint
                    # (the checkpoint may have different classes than COCO_SUPERCATEGORIES)
                    from src.utils.downloader import get_model_path
                    checkpoint_path = get_model_path(model_type, model_loader.config)
                    if checkpoint_path:
                        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
                        if isinstance(ckpt, dict) and 'class_names' in ckpt:
                            class_names = list(ckpt['class_names'])

                    # Process image
                    inputs = processor(images=image, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(device)
                    st.info(f"Processed Tensor Shape: {list(pixel_values.shape)} (B x C x H x W)")

                    logits = model(pixel_values, text=text_caption if text_caption else None)
                    num_logits = logits.shape[-1]
                    k = min(len(class_names), num_logits)
                    predictions = get_topk_predictions(logits, class_names[:num_logits], k=k)

                # Display results
                st.subheader(f"Predictions ({len(class_names)} categories)")
                for label, score in predictions:
                    st.progress(float(score), text=f"{label}: {score:.3f}")

                with st.expander("See detailed scores"):
                    for label, score in predictions:
                        st.write(f"**{label}**: {score:.4f}")

            except Exception as e:
                st.error(f"CLIP inference failed: {e}")
                logger.exception("CLIP inference error")


# =====================================================================
# Main Application
# =====================================================================

def main():
    """Main Streamlit application."""
    st.title("🧠 Deep Learning Model Inference")
    st.markdown("""
    This app demonstrates inference for models from **DL Assignment 1**:
    - **Image Classification**: CNN (ResNet-50) vs Vision Transformer (ViT) on CIFAR-100
    - **Text Classification**: BiLSTM (GloVe frozen) vs BERT (frozen backbone) on DBpedia-14
    - **Multimodal Classification**: CLIP Zero-Shot vs Few-Shot (Prototype Network) on COCO

    ### Instructions:
    1. Select a tab for the task you want to test
    2. Upload an image or enter text
    3. Choose a model (models will be downloaded automatically on first use)
    4. Click the inference button to see predictions
    """)

    # Setup sidebar
    device, force_download = setup_sidebar()

    # Initialize model loader
    model_loader = get_model_loader()

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "🖼 Image Classification",
        "📝 Text Classification",
        "🔗 Multimodal (CLIP)"
    ])

    with tab1:
        image_classification_tab(model_loader, device)

    with tab2:
        text_classification_tab(model_loader, device)

    with tab3:
        multimodal_classification_tab(model_loader, device)

    # Footer
    st.markdown("---")
    st.caption("""
    **Note**: Models are cached in `~/.cache/dl-assignment1/models/`.
    First-time use will download models from Google Drive (requires internet connection).
    CLIP Zero-Shot does not require any checkpoint — it uses the pretrained model directly.
    """)

    # Debug info (collapsed)
    with st.expander("Debug Information"):
        st.write(f"Device: {device}")
        st.write(f"Torch CUDA available: {torch.cuda.is_available()}")
        st.write(f"Cache directory: {Path.home() / '.cache' / 'dl-assignment1' / 'models'}")

if __name__ == "__main__":
    main()