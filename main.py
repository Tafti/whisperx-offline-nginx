import tempfile
import os
from typing import Optional
import threading
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import whisperx

# Import configuration and utilities
from config import logger, device, compute_type, MODEL_CACHE_ROOT
from utils import load_model

# ---------- Global model references ----------
whisper_model = None
alignment_model_cache = {}
alignment_cache_lock = threading.Lock()

# ---------- FastAPI app ----------
app = FastAPI(title="WhisperX API", description="Local transcription server with retry logic")

@app.on_event("startup")
def load_models():
    global whisper_model
    # ! change model here
    # TODO: maybe add to /transcribe parameters
    model_name = "large-v3"
    logger.info(f"Loading Whisper ASR model: {model_name}")
    whisper_model = load_model(model_name, device, compute_type)
    logger.info("ASR model loaded successfully")


def get_alignment_model(language_code: str):
    """Get cached alignment model for language or load it once."""
    with alignment_cache_lock:
        cached = alignment_model_cache.get(language_code)
        if cached is not None:
            return cached

    try:
        logger.info(f"Loading alignment model for language: {language_code}")
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language_code,
            device=device,
            model_cache_only=False,
        )
        with alignment_cache_lock:
            alignment_model_cache[language_code] = (align_model, align_metadata)
        logger.info(f"Alignment model loaded and cached: {language_code}")
        return align_model, align_metadata
    except Exception as e:
        logger.warning(f"Alignment model unavailable for '{language_code}': {e}")
        return None, None

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = "en",
    beam_size: int = 5,
    best_of: int = 5,
    temperature: float = 0.0
):
    """Transcribe an audio file using faster-whisper."""
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
        raise HTTPException(400, "Unsupported file format. Use .wav, .mp3, .m4a, .ogg, or .flac")

    # Save uploaded file temporarily
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Transcribe using faster-whisper (no batch_size parameter)
        segments, info = whisper_model.transcribe(
            tmp_path,
            language=language,
            task="transcribe",
            beam_size=beam_size,
            best_of=best_of,
            temperature=temperature,
            vad_filter=True,  # Enable VAD for better silence detection
            vad_parameters={
                "min_silence_duration_ms": 500,
                "threshold": 0.5
            }
        )
        
        # Collect segments
        transcript_segments = []
        full_text = []
        
        for segment in segments:
            transcript_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })
            full_text.append(segment.text.strip())
        
        result = {
            "language": info.language,
            "language_probability": info.language_probability,
            "text": " ".join(full_text),
            "segments": transcript_segments
        }
        
        # Load alignment model per detected language and cache it for reuse.
        detected_language = (info.language or "").strip().lower()
        if detected_language:
            try:
                align_model, align_metadata = get_alignment_model(detected_language)

                if align_model is None:
                    return JSONResponse(content=result)

                # Convert to format expected by whisperx.align
                aligned = whisperx.align(
                    transcript_segments,
                    align_model,
                    align_metadata,
                    tmp_path,
                    device
                )
                result["word_segments"] = aligned["segments"]
            except Exception as e:
                logger.warning(f"Alignment failed: {e}")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(500, f"Transcription error: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "device": device,
        "model_loaded": whisper_model is not None,
        "cache_dir": str(MODEL_CACHE_ROOT)
    }

@app.get("/")
async def root():
    return {"message": "WhisperX API is running. Use POST /transcribe to transcribe audio."}