import os
import shutil
import re
from pathlib import Path
import whisperx
from faster_whisper import WhisperModel
from config import MODEL_CACHE_ROOT, logger

def find_model_snapshot_path(model_name: str):
    """
    Find the actual snapshot path of a downloaded faster-whisper model.
    Handles Hugging Face's symlink-based cache structure.
    """
    # Common patterns for faster-whisper cache
    search_patterns = [
        MODEL_CACHE_ROOT / "hub" / f"models--guillaumekln--faster-whisper-{model_name}" / "snapshots",
        MODEL_CACHE_ROOT / f"models--guillaumekln--faster-whisper-{model_name}" / "snapshots",
        MODEL_CACHE_ROOT / "hub" / f"models--Systran--faster-whisper-{model_name}" / "snapshots",
    ]
    
    for pattern in search_patterns:
        if pattern.exists():
            logger.info(f"Checking pattern: {pattern}")
            # Find all snapshot directories
            snapshots = [d for d in pattern.iterdir() if d.is_dir()]
            for snapshot in snapshots:
                # Check if model.bin exists (as file OR symlink)
                model_bin = snapshot / "model.bin"
                if model_bin.exists() or model_bin.is_symlink():
                    # If it's a symlink, resolve to actual path
                    if model_bin.is_symlink():
                        resolved_path = model_bin.resolve()
                        logger.info(f"Found symlink: {model_bin} -> {resolved_path}")
                    else:
                        logger.info(f"Found file: {model_bin}")
                    
                    # The snapshot directory contains the model files structure
                    logger.info(f"Snapshot directory: {snapshot}")
                    return str(snapshot)
    
    # Alternative: search for any directory containing model.bin (including symlinks)
    logger.info("Searching recursively for model.bin (including symlinks)...")
    for model_bin in MODEL_CACHE_ROOT.rglob("model.bin"):
        if model_bin.exists() or model_bin.is_symlink():
            snapshot_path = model_bin.parent
            logger.info(f"Found model.bin at: {model_bin}")
            logger.info(f"Parent directory: {snapshot_path}")
            return str(snapshot_path)
    
    return None

def safe_load_model(model_name: str, device: str, compute_type: str, max_retries: int = 2):
    """
    Load model directly using faster-whisper with explicit path.
    Handles Hugging Face's symlink cache structure.
    """
    # First, try to find the local model path
    model_path = find_model_snapshot_path(model_name)
    
    if model_path:
        logger.info(f"Loading model directly from snapshot: {model_path}")
        
        # Verify that essential files exist (considering symlinks)
        model_bin = Path(model_path) / "model.bin"
        if not (model_bin.exists() or model_bin.is_symlink()):
            raise FileNotFoundError(f"model.bin not found in {model_path}")
        
        try:
            # Use faster_whisper directly with snapshot path
            model = WhisperModel(
                model_path,  # Snapshot directory path
                device=device,
                compute_type=compute_type,
                cpu_threads=4,
                num_workers=1
            )
            logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            
            # Try loading from parent directory as fallback
            parent_path = Path(model_path).parent.parent
            logger.info(f"Attempting fallback: {parent_path}")
            try:
                model = WhisperModel(
                    str(parent_path),
                    device=device,
                    compute_type=compute_type
                )
                logger.info(f"Model loaded from fallback path: {parent_path}")
                return model
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                raise
    
    # If we can't find the model, list what we do have
    logger.error(f"Model '{model_name}' not found in {MODEL_CACHE_ROOT}")
    logger.info("Available model directories:")
    for path in MODEL_CACHE_ROOT.rglob("snapshots"):
        if path.is_dir():
            logger.info(f"  - {path}")
    
    raise FileNotFoundError(
        f"Model '{model_name}' not found in {MODEL_CACHE_ROOT}\n"
        f"Model is expected to be at: {MODEL_CACHE_ROOT}/hub/models--guillaumekln--faster-whisper-{model_name}/snapshots/*/\n"
        f"But we found it at: ./data/models/models--guillaumekln--faster-whisper-base/snapshots/515102184abb526d1cfb9c882107192588d7250a/\n"
        f"Please check the path structure."
    )