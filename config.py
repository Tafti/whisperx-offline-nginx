import os
import logging
from pathlib import Path


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

# ---------- Paths ----------
BASE_DIR = Path(__file__).parent
load_env_file(BASE_DIR / ".env")
MODEL_CACHE_ROOT = BASE_DIR / "data" / "models"
MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# Set environment variables for Hugging Face and WhisperX caching
os.environ["HF_HOME"] = str(MODEL_CACHE_ROOT)
os.environ["TRANSFORMERS_CACHE"] = str(MODEL_CACHE_ROOT / "transformers")
os.environ["WHISPERX_CACHE"] = str(MODEL_CACHE_ROOT / "whisperx")

# CRITICAL: Force offline mode - prevents any network requests
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

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
logger.info(f"Offline mode: {'ENABLED' if os.getenv('HF_HUB_OFFLINE') == '1' else 'DISABLED'} (HF_HUB_OFFLINE={os.getenv('HF_HUB_OFFLINE')})")