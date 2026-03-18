"""Streamlit inference app for DL Assignment 1 models."""

import streamlit as st
import torch
from PIL import Image
from pathlib import Path
import sys
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.inference_utils import (
    ModelLoader, preprocess_image, get_topk_predictions,
    preprocess_text_rnn, preprocess_text_transformers
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
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image")

    if image_file and model_type and st.button("Run Inference", type="primary", key="image_infer"):
        with st.spinner("Loading model and running inference..."):
            try:
                # Load model
                model = model_loader.load_model(model_type, device=device)
                class_names = model_loader.get_class_names(model_type)

                # Preprocess and predict
                tensor = preprocess_image(image, device=device)
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

def text_classification_tab(model_loader, device):
    """Text classification tab."""
    st.header("📝 Text Classification (DBpedia-14)")

    text_input = st.text_area(
        "Enter text for classification:",
        height=150,
        placeholder="Enter a news article or description..."
    )

    model_type = st.selectbox(
        "Select Model",
        ["text.bilstm", "text.distilbert"],
        key="text_model"
    )

    if text_input and model_type and st.button("Classify Text", type="primary", key="text_infer"):
        with st.spinner("Processing text..."):
            try:
                # Load model
                model = model_loader.load_model(model_type, device=device)
                class_names = model_loader.get_class_names(model_type)

                # Text preprocessing depends on model type
                if model_type == "text.bilstm":
                    # Load vocab from GDrive
                    vocab_path = model_loader.get_auxiliary_path(model_type, "vocab_id")
                    if not vocab_path:
                        st.error("No vocabulary file (vocab_id) defined for BiLSTM in config.")
                        return
                    
                    with open(vocab_path, "rb") as f:
                        vocab = pickle.load(f)
                    
                    tensor = preprocess_text_rnn(text_input, vocab).to(device)
                    logits = model(tensor)
                else:  # distilbert
                    # Load tokenizer from HF (architecture-only, use model_name from config)
                    model_info = model_loader.get_model_info(model_type)
                    model_name = model_info.get("model_name", "distilbert-base-uncased")
                    # tokenizer = AutoTokenizer.from_pretrained(model_name)
                    
                    # encoded = preprocess_text_transformers(text_input, tokenizer)
                    # encoded = {k: v.to(device) for k, v in encoded.items()}
                    # logits = model(**encoded)
                    pass

                predictions = get_topk_predictions(logits, class_names, k=5)

                # Display results
                st.subheader("Top-5 Predictions")
                for label, score in predictions:
                    st.progress(float(score), text=f"{label}: {score:.3f}")

            except Exception as e:
                st.error(f"Text classification failed: {e}")
                logger.exception("Text inference error")

def multimodal_classification_tab(model_loader, device):
    """Multimodal classification tab."""
    st.header("🔗 Multimodal Classification (CLIP)")

    col1, col2 = st.columns([1, 1])

    with col1:
        clip_image = st.file_uploader(
            "Upload image for CLIP classification",
            type=["jpg", "jpeg", "png"],
            key="clip_upload"
        )

        model_type = st.selectbox(
            "Select CLIP Model",
            ["multimodal.clip_zero", "multimodal.clip_few"],
            key="clip_model"
        )

    with col2:
        if clip_image:
            image = Image.open(clip_image)
            st.image(image, caption="Uploaded Image")

    if clip_image and model_type and st.button("Run CLIP Inference", type="primary", key="clip_infer"):
        with st.spinner("Running CLIP inference..."):
            try:
                model = model_loader.load_model(model_type, device=device)
                class_names = model_loader.get_class_names(model_type)

                # CLIP-specific preprocessing
                if hasattr(model, 'get_preprocess'):
                    preprocess = model.get_preprocess()
                    tensor = preprocess(image).unsqueeze(0).to(device)
                else:
                    # Fallback to standard preprocessing
                    tensor = preprocess_image(image, device=device)

                with torch.no_grad():
                    if model_type == "multimodal.clip_zero":
                        # Zero-shot classification: encode prompts first
                        prompts = [f"a photo of a {c}" for c in class_names]
                        model.encode_text_prompts(prompts, device=device)
                        
                        _, probs = model.predict(tensor)
                        logits = torch.log(probs + 1e-10)  # Convert to logits-like
                    else:
                        # Few-shot classification (linear probe)
                        logits = model(tensor)

                predictions = get_topk_predictions(logits, class_names, k=10)

                st.subheader("Top-10 Predictions")
                for label, score in predictions:
                    st.progress(float(score), text=f"{label}: {score:.3f}")

            except Exception as e:
                st.error(f"CLIP inference failed: {e}")
                logger.exception("CLIP inference error")

def main():
    """Main Streamlit application."""
    st.title("🧠 Deep Learning Model Inference")
    st.markdown("""
    This app demonstrates inference for models from **DL Assignment 1**:
    - **Image Classification**: CNN (ResNet-50) vs Vision Transformer (ViT) on CIFAR-100
    - **Text Classification**: BiLSTM vs DistilBERT on DBpedia-14
    - **Multimodal Classification**: CLIP Zero-shot vs Few-shot on CIFAR-100

    ### Instructions:
    1. Select a tab for the task you want to test
    2. Upload an image or enter text
    3. Choose a model (models will be downloaded automatically on first use)
    4. Click "Run Inference" to see predictions
    """)

    # Setup sidebar
    device, force_download = setup_sidebar()

    # Initialize model loader
    model_loader = get_model_loader()

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "🖼 Image Classification",
        "📝 Text Classification",
        "🔗 Multimodal"
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
    """)

    # Debug info (collapsed)
    with st.expander("Debug Information"):
        st.write(f"Device: {device}")
        st.write(f"Torch CUDA available: {torch.cuda.is_available()}")
        st.write(f"Cache directory: {Path.home() / '.cache' / 'dl-assignment1' / 'models'}")

if __name__ == "__main__":
    main()