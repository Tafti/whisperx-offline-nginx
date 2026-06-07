from faster_whisper import WhisperModel
from config import logger

def load_model(model_name: str, device: str, compute_type: str):
    """Load a faster-whisper model by name using default Hugging Face cache resolution."""
    logger.info(f"Loading model from Hugging Face cache by name: {model_name}")
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=4,
        num_workers=1,
    )
    logger.info(f"Model loaded successfully: {model_name}")
    return model