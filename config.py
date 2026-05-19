import os
import logging
from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).parent
MODEL_CACHE_ROOT = BASE_DIR / "data" / "models"
MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# Set environment variables for Hugging Face and WhisperX caching
os.environ["HF_HOME"] = str(MODEL_CACHE_ROOT)
os.environ["TRANSFORMERS_CACHE"] = str(MODEL_CACHE_ROOT / "transformers")
os.environ["WHISPERX_CACHE"] = str(MODEL_CACHE_ROOT / "whisperx")

# CRITICAL: Force offline mode - prevents any network requests
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Disable telemetry and reduce timeouts
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("whisperx_api")

# ---------- Device ----------
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

logger.info(f"Using device: {device}, compute type: {compute_type}")
logger.info(f"Model cache root: {MODEL_CACHE_ROOT}")
logger.info(f"Offline mode: ENABLED (HF_HUB_OFFLINE=1)")