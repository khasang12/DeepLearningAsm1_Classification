import os
import gdown
import hashlib
import json
from pathlib import Path
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger("downloader")

# Cache directory for downloaded models
CACHE_DIR = Path.home() / ".cache" / "dl-assignment1" / "models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def download_from_gdrive(url_or_id: str, output_path: str, quiet: bool = False) -> Optional[str]:
    """
    Download a file from Google Drive.
    
    Args:
        url_or_id: Google Drive file URL or ID.
        output_path: Path to save the downloaded file.
        quiet: If True, suppress output.
        
    Returns:
        The path to the downloaded file, or None if failed.
    """
    if os.path.exists(output_path):
        logger.info(f"File already exists at {output_path}. Skipping download.")
        return output_path
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        logger.info(f"Downloading from Google Drive: {url_or_id}")
        # gdown can handle both full URL and file ID
        path = gdown.download(url=url_or_id, output=output_path, quiet=quiet, fuzzy=True)
        if path:
            logger.info(f"Successfully downloaded to {output_path}")
            return path
        else:
            logger.error(f"Download failed for {url_or_id}")
            return None
    except Exception as e:
        logger.error(f"Error downloading from Google Drive: {e}")
        return None


def download_model(
    file_id: str,
    output_name: str,
    force: bool = False,
    quiet: bool = True
) -> Path:
    """Download model from Google Drive with caching.

    Parameters
    ----------
    file_id : str
        Google Drive file ID (from shareable link)
    output_name : str
        Name for the cached file
    force : bool, default=False
        Force re-download even if cached
    quiet : bool, default=True
        Suppress gdown output

    Returns
    -------
    Path
        Path to downloaded/cached model file
    """
    cache_path = CACHE_DIR / output_name

    if cache_path.exists() and not force:
        logger.debug(f"Using cached model: {cache_path}")
        return cache_path

    logger.info(f"Downloading model: {output_name}")
    
    # Check if file_id is already a URL
    if file_id.startswith("http"):
        url = file_id
    else:
        url = f"https://drive.google.com/uc?id={file_id}"

    try:
        gdown.download(url, str(cache_path), quiet=quiet, fuzzy=True)
        
        # Validation check: if it's an HTML file, it's likely a GDrive error page
        with open(cache_path, 'rb') as f:
            chunk = f.read(100)
            if b'<!DOCTYPE html>' in chunk or b'<html' in chunk:
                cache_path.unlink()
                raise ValueError("Downloaded file is an HTML page (likely GDrive download limit or scan warning). "
                                 "Check if the file is shared publicly and try again.")
        
        logger.info(f"Downloaded to: {cache_path}")
    except Exception as e:
        logger.error(f"Failed to download {file_id}: {e}")
        if cache_path.exists():
            logger.warning(f"Using existing cached file despite error: {cache_path}")
        else:
            raise

    return cache_path


def get_model_path(
    model_key: str,
    models_config: dict,
    force_download: bool = False
) -> Optional[Path]:
    """Get local path to model checkpoint, downloading if needed.

    Parameters
    ----------
    model_key : str
        Model key in config (e.g., "image.cnn")
    models_config : dict
        Loaded models.json configuration
    force_download : bool, default=False
        Force re-download even if cached

    Returns
    -------
    Optional[Path]
        Path to model checkpoint, or None if no checkpoint_id specified
    """
    # Navigate nested config
    parts = model_key.split(".")
    model_info = models_config
    for part in parts:
        model_info = model_info.get(part, {})

    if not model_info or "checkpoint_id" not in model_info:
        return None

    file_id = model_info["checkpoint_id"]
    # Create cache name with hash of file_id for uniqueness
    cache_name = f"{model_key.replace('.', '_')}_{hashlib.md5(file_id.encode()).hexdigest()[:8]}.pth"

    return download_model(file_id, cache_name, force=force_download)


def clear_cache(model_key: str = None) -> int:
    """Clear cached models.

    Parameters
    ----------
    model_key : str, optional
        Specific model key to clear (e.g., "image.cnn"), or clear all if None

    Returns
    -------
    int
        Number of files deleted
    """
    if model_key is None:
        # Clear entire cache directory (.pt and .pth files)
        count = 0
        for ext in ("*.pt", "*.pth"):
            for file in CACHE_DIR.glob(ext):
                file.unlink()
                count += 1
        logger.info(f"Cleared all {count} cached models")
        return count
    else:
        # Clear specific model pattern (.pt and .pth files)
        base_pattern = f"{model_key.replace('.', '_')}_*"
        count = 0
        for ext in (".pt", ".pth"):
            pattern = f"{base_pattern}{ext}"
            for file in CACHE_DIR.glob(pattern):
                file.unlink()
                count += 1
        logger.info(f"Cleared {count} cached files for {model_key}")
        return count
