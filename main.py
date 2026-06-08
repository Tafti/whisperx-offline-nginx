import tempfile
import os
import asyncio
import threading
import zipfile
from pathlib import Path
from typing import Optional, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import whisperx

# Import configuration and utilities
from config import logger, device, compute_type, MODEL_CACHE_ROOT
from utils import load_model

# ---------- Global model references ----------
whisper_model = None
MAX_CONCURRENT_TRANSCRIPTIONS = int(os.getenv("MAX_CONCURRENT_TRANSCRIPTIONS", "2"))
transcription_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIPTIONS)
ENABLE_ALIGNMENT = os.getenv("ENABLE_ALIGNMENT", "1") == "1"
ALIGN_MODEL_DIR = os.getenv("ALIGN_MODEL_DIR", str(MODEL_CACHE_ROOT / "hub"))
ALIGN_MODEL_PATHS = {
    "fa": os.getenv(
        "ALIGN_MODEL_FA",
        str(
            MODEL_CACHE_ROOT
            / "hub"
            / "models--jonatasgrosman--wav2vec2-large-xlsr-53-persian"
        ),
    )
}

_align_models = {}
_align_models_lock = threading.Lock()
_sentence_splitter_patched = False
_sentence_splitter_force_fallback = False
_sentence_splitter_warned = False


def _prepare_nltk_resources() -> None:
    """Repair common broken NLTK punkt archives and try to bootstrap tokenizer data."""
    try:
        import nltk
    except Exception as exc:
        logger.warning("NLTK import failed: %s", exc)
        return

    # Remove corrupted punkt archives that trigger BadZipFile during alignment.
    for base_dir in nltk.data.path:
        tokenizers_dir = os.path.join(base_dir, "tokenizers")
        for archive_name in ("punkt_tab.zip", "punkt.zip"):
            archive_path = os.path.join(tokenizers_dir, archive_name)
            if not os.path.exists(archive_path):
                continue
            try:
                with zipfile.ZipFile(archive_path):
                    pass
            except Exception:
                try:
                    os.remove(archive_path)
                    logger.info("Removed corrupted NLTK archive: %s", archive_path)
                except OSError as remove_exc:
                    logger.warning("Failed to remove corrupted archive %s: %s", archive_path, remove_exc)

    # Try to ensure sentence tokenizers exist. This can fail in strict offline mode; runtime fallback handles that.
    try:
        nltk.data.load("tokenizers/punkt_tab/english.pickle")
        return
    except Exception:
        pass

    try:
        nltk.download("punkt_tab", quiet=True)
        nltk.data.load("tokenizers/punkt_tab/english.pickle")
        logger.info("NLTK punkt_tab tokenizer is available")
        return
    except Exception:
        pass

    try:
        nltk.download("punkt", quiet=True)
        nltk.data.load("tokenizers/punkt/english.pickle")
        logger.info("Using legacy NLTK punkt tokenizer")
    except Exception:
        logger.warning("NLTK punkt resources unavailable; alignment will use regex fallback splitter")

# ---------- FastAPI app ----------
app = FastAPI(title="WhisperX API", description="Local transcription server with retry logic")


def _transcribe_blocking(
    tmp_path: str,
    language: Optional[str],
    beam_size: int,
    best_of: int,
    temperature: float
) -> Tuple[list, object]:
    """Run transcription and materialize generator off the event loop."""
    segments, info = whisper_model.transcribe(
        tmp_path,
        language=language,
        task="transcribe",
        beam_size=beam_size,
        best_of=best_of,
        temperature=temperature,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "threshold": 0.5
        }
    )
    return list(segments), info


def _patch_alignment_sentence_splitter() -> None:
    """Fallback to a simple splitter when NLTK punkt resources are missing/corrupted."""
    global _sentence_splitter_patched, _sentence_splitter_force_fallback, _sentence_splitter_warned
    if _sentence_splitter_patched:
        return

    try:
        import whisperx.alignment as wx_alignment
    except Exception as exc:
        logger.warning("Unable to import whisperx.alignment for patching: %s", exc)
        return

    original_nltk_load = wx_alignment.nltk_load

    class FallbackSentenceSplitter:
        def span_tokenize(self, text):
            return [(0, len(text))] if text else []

    fallback_splitter = FallbackSentenceSplitter()

    def safe_nltk_load(resource_name):
        global _sentence_splitter_force_fallback, _sentence_splitter_warned
        if _sentence_splitter_force_fallback:
            return fallback_splitter

        try:
            return original_nltk_load(resource_name)
        except Exception:
            # Prefer legacy punkt tokenizer if punkt_tab is unavailable.
            try:
                if resource_name.startswith("tokenizers/punkt_tab/"):
                    legacy_resource = resource_name.replace("tokenizers/punkt_tab/", "tokenizers/punkt/")
                    return original_nltk_load(legacy_resource)
            except Exception:
                pass

            _sentence_splitter_force_fallback = True
            if not _sentence_splitter_warned:
                logger.warning(
                    "NLTK sentence tokenizer unavailable; using regex fallback sentence splitter."
                )
                _sentence_splitter_warned = True
            return fallback_splitter

    wx_alignment.nltk_load = safe_nltk_load
    _sentence_splitter_patched = True


def _resolve_align_model_name(language_code: str) -> Optional[str]:
    configured = ALIGN_MODEL_PATHS.get(language_code)
    if not configured:
        return None

    configured_path = Path(configured)
    if not configured_path.exists():
        return None

    # If the path already points to a snapshot directory, use it directly.
    if (configured_path / "model.safetensors").exists() or (configured_path / "pytorch_model.bin").exists():
        return str(configured_path)

    # Hugging Face cache layout: models--.../refs/main -> snapshots/<commit_hash>
    refs_main = configured_path / "refs" / "main"
    snapshots_dir = configured_path / "snapshots"
    if refs_main.exists() and snapshots_dir.exists():
        try:
            revision = refs_main.read_text(encoding="utf-8").strip()
            snapshot_dir = snapshots_dir / revision
            if snapshot_dir.exists():
                return str(snapshot_dir)
        except OSError:
            pass

    # Fallback: choose an available snapshot if refs/main is missing.
    if snapshots_dir.exists():
        snapshots = sorted((p for p in snapshots_dir.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
        if snapshots:
            return str(snapshots[0])

    return None


def _get_or_load_align_model(language_code: str):
    with _align_models_lock:
        if language_code in _align_models:
            return _align_models[language_code]

    model_name = _resolve_align_model_name(language_code)
    if model_name is None:
        logger.warning("No local alignment model configured for language: %s", language_code)
        return None

    logger.info("Loading alignment model for language %s from %s", language_code, model_name)
    align_model, align_metadata = whisperx.load_align_model(
        language_code=language_code,
        device=device,
        model_name=model_name,
        model_dir=ALIGN_MODEL_DIR,
        model_cache_only=True,
    )

    with _align_models_lock:
        _align_models[language_code] = (align_model, align_metadata)
    logger.info("Alignment model loaded for language %s", language_code)
    return align_model, align_metadata


def _align_words_blocking(transcript_segments: list, audio_path: str, language_code: str) -> list:
    loaded = _get_or_load_align_model(language_code)
    if loaded is None:
        return []

    align_model, align_metadata = loaded
    aligned = whisperx.align(
        transcript_segments,
        align_model,
        align_metadata,
        audio_path,
        device,
    )

    word_segments = []
    for item in aligned.get("word_segments", []):
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            continue
        word_segments.append(
            {
                "word": (item.get("word") or "").strip(),
                "start": round(float(start), 2),
                "end": round(float(end), 2),
                "score": round(float(item["score"]), 4) if item.get("score") is not None else None,
            }
        )

    return word_segments

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
    language: str = None,          # ← None enables auto‑detection
    beam_size: int = 5,
    best_of: int = 5,
    temperature: float = 0.0
):
    """Transcribe an audio file with automatic language detection."""
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
        raise HTTPException(400, "Unsupported file format. Use .wav, .mp3, .m4a, .ogg, or .flac")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if whisper_model is None:
            raise HTTPException(503, "ASR model is not loaded")

        # Limit concurrent heavy jobs while still allowing concurrent HTTP handling.
        async with transcription_semaphore:
            segments, info = await asyncio.to_thread(
                _transcribe_blocking,
                tmp_path,
                language,
                beam_size,
                best_of,
                temperature,
            )
        
        # Collect segments
        transcript_segments = []
        raw_word_segments = []
        full_text = []
        for segment in segments:
            transcript_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })

            # Fallback word-level timestamps from faster-whisper output.
            if getattr(segment, "words", None):
                for word in segment.words:
                    if word.start is None or word.end is None:
                        continue
                    raw_word_segments.append({
                        "word": word.word.strip() if word.word else "",
                        "start": round(word.start, 2),
                        "end": round(word.end, 2),
                        "score": round(float(word.probability), 4) if word.probability is not None else None
                    })

            full_text.append(segment.text.strip())
        
        aligned_word_segments = []
        if ENABLE_ALIGNMENT:
            try:
                aligned_word_segments = await asyncio.to_thread(
                    _align_words_blocking,
                    transcript_segments,
                    tmp_path,
                    info.language,
                )
            except Exception as exc:
                logger.warning("Alignment failed for language %s: %s", info.language, exc)

        word_segments = aligned_word_segments if aligned_word_segments else raw_word_segments
        if aligned_word_segments:
            word_segments_source = "wav2vec_alignment"
        elif raw_word_segments:
            word_segments_source = "asr_raw"
        else:
            word_segments_source = None

        result = {
            "language": info.language,
            "language_probability": info.language_probability,
            "text": " ".join(full_text),
            "segments": transcript_segments,
            "word_segments": word_segments,
            "word_segments_source": word_segments_source,
        }
        
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
        "alignment_enabled": ENABLE_ALIGNMENT,
        "loaded_alignment_languages": sorted(list(_align_models.keys())),
        "max_concurrent_transcriptions": MAX_CONCURRENT_TRANSCRIPTIONS,
        "cache_dir": str(MODEL_CACHE_ROOT)
    }

@app.get("/")
async def root():
    return {"message": "WhisperX API is running. Use POST /transcribe to transcribe audio."}