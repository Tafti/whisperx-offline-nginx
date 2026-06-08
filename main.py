import tempfile
import os
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import whisperx

# Import configuration and utilities
from config import logger, device, compute_type, MODEL_CACHE_ROOT
from utils import load_model

# ---------- Global model references ----------
whisper_model = None
align_model = None
align_metadata = None

# ---------- FastAPI app ----------
app = FastAPI(title="WhisperX API", description="Local transcription server with retry logic")

@app.on_event("startup")
def load_models():
    global whisper_model, align_model, align_metadata
    # ! change model here
    # TODO: maybe add to /transcribe parameters
    model_name = "large-v3"
    logger.info(f"Loading Whisper ASR model: {model_name}")
    whisper_model = load_model(model_name, device, compute_type)
    logger.info("ASR model loaded successfully")

    # Attempt to load alignment model, but continue if it fails
    try:
        logger.info("Loading alignment model...")
        align_model, align_metadata = whisperx.load_align_model(
            language_code="en", 
            device=device,
            model_cache_only=True 
        )
        logger.info("Alignment model loaded successfully")
    except Exception as e:
        logger.warning(f"Alignment model not available: {e}")
        logger.warning("Continuing without word-level alignment")
        align_model = None
        align_metadata = None

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = None,
    beam_size: int = 5,
    best_of: int = 5,
    temperature: float = 0.0
):
    """Transcribe an audio file using faster-whisper."""
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
        raise HTTPException(400, "Unsupported file format. Use .wav, .mp3, .m4a, .ogg, or .flac")

    # Stream upload to disk in chunks to avoid loading large files into memory.
    chunk_size = 1024 * 1024  # 1 MB
    suffix = os.path.splitext(file.filename)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        # Keep auto-detection enabled unless a language was explicitly provided.
        selected_language = language.strip() if language else None
        selected_language = selected_language or None

        # Transcribe using faster-whisper (no batch_size parameter)
        segments, info = whisper_model.transcribe(
            tmp_path,
            language=selected_language,
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
        
        # If alignment model is available, add word-level timestamps
        if align_model is not None:
            try:
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
        "device": device
    }

@app.get("/")
async def root():
    return {"message": "Transcription API is running. Use POST /transcribe to transcribe audio."}