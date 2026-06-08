# 🎙️ WhisperX Local API

**Self-hosted, offline speech-to-text API with word‑level timestamps**  
Built with FastAPI + WhisperX (faster-whisper) – fully local, no cloud dependencies.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![WhisperX](https://img.shields.io/badge/WhisperX-3.1.1-8A2BE2)](https://github.com/m-bain/whisperX)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://docker.com)

---

## ✨ Features

- 🚀 **High performance** – Uses `faster-whisper` (CTranslate2) for CPU/GPU acceleration.
- 🎯 **Word‑level timestamps** – Optional forced alignment via wav2vec 2.0.
- 🔒 **Offline capable** – No internet required after first model download.
- 🐳 **Docker ready** – `docker-compose up` to start.
- 💾 **Model caching** – Models stored locally in `./data/models`; survives container restarts.
- 🔁 **Resumable downloads** – Retries and cleans corrupted snapshots.
- 📦 **Multiple models** – `base`, `small`, `medium`, `large-v2`, `large-v3`.
- 🧪 **Health endpoint** – Monitor server status.
- 🛡️ **Privacy first** – Audio never leaves your server.

---

## 📋 Prerequisites

- **Python** 3.12+ (if running locally)
- **FFmpeg** (required for audio decoding)
- **Docker** and **Docker Compose** (optional, for containerised deployment)

### Install FFmpeg

| OS | Command |
|----|---------|
| Ubuntu/Debian | `sudo apt install ffmpeg` |
| macOS (Homebrew) | `brew install ffmpeg` |
| Windows | Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add to `PATH` |

---

## 🚀 Quick Start

### Option 1: Docker (recommended)

```bash
git clone https://github.com/yourusername/whisperx-local.git
cd whisperx-local
docker compose up --build
```

The API will be available at `http://localhost:8000`.

### Option 2: Local Python

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🎯 Usage

### Transcribe an audio file

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@/path/to/audio.mp3" \
  -F "beam_size=5"
```

If you omit `language`, WhisperX auto-detects it. You can still force a language by passing `-F "language=fr"` (or any supported code).

**Response example** (without alignment):
```json
{
  "language": "en",
  "language_probability": 0.98,
  "text": "Hello world, this is a test.",
  "segments": [
    {"start": 0.0, "end": 1.2, "text": "Hello world,"},
    {"start": 1.2, "end": 2.5, "text": "this is a test."}
  ]
}
```

### Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "device": "cpu",
  "model_loaded": true,
  "cache_dir": "/app/data/models"
}
```

### Interactive docs

Open `http://localhost:8000/docs` for Swagger UI.

---

## ⚙️ Configuration

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_HOME` | Hugging Face cache root | `./data/models` |
| `HF_HOME` | Transformers cache | `./data/models/transformers` |
| `HF_HUB_OFFLINE` | Force offline mode | `0` (can be set to `1`) |
| `WHISPERX_CACHE` | WhisperX specific cache | `./data/models/whisperx` |

### Model selection

In `main.py`, change the `model_name` variable:
```python
model_name = "base"  # options: base, small, medium, large-v2, large-v3
```

### Alignment model (word‑level timestamps)

Enable by setting `align_model` in the startup routine. If alignment fails, the server continues without it. To force offline usage:

```python
align_model, align_metadata = whisperx.load_align_model(
    language_code="en",
    device=device,
    model_cache_only=True
)
```

---

## 🐳 Docker Details

### Build and run

```bash
docker compose up --build
```

### Stop and remove

```bash
docker compose down
```

### Persist models

Models are stored in `./data/models` on your host. This volume is mounted into the container, so models survive container recreation.

### GPU support (optional)

Uncomment the `deploy` section in `docker-compose.yml` and use the GPU‑enabled Dockerfile (provided in the repo as `Dockerfile.gpu`).

---

## 🧠 How It Works

1. **ASR** – `faster-whisper` loads a CTranslate2‑converted Whisper model from local cache.
2. **Alignment** (optional) – wav2vec 2.0 model refines word boundaries.
3. **FastAPI** – Exposes endpoints, handles file uploads, and streams JSON responses.

Model download occurs automatically on first run, and subsequent runs reuse the local Hugging Face cache.

---

## ❓ Troubleshooting

### 1. SSL / Connection errors when loading alignment model
**Solution:** Set `model_cache_only=True` and ensure the model was downloaded once while online. Or skip alignment entirely – the API still transcribes.

### 2. `load_align_model() got an unexpected keyword argument 'local_files_only'`
**Solution:** Use `model_cache_only=True` instead (as shown above).

### 3. Model download fails with `ConnectionResetError`
**Solution:** Use the manual download script (`download_model.py`) or set environment variable `HF_ENDPOINT=https://hf-mirror.com`.

### 4. FFmpeg not found
**Solution:** Install FFmpeg (see Prerequisites). In Docker, it is already included.

### 5. GPU not detected inside container
**Solution:** Install `nvidia-docker` and uncomment the GPU section in `docker-compose.yml`.

---

## 📁 Project Structure

```
whisperx-local/
├── main.py                 # FastAPI endpoints
├── config.py               # Paths, logging, device detection
├── utils/
│   ├── __init__.py
│   └── model_loader.py     # Default faster-whisper model loader
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── .dockerignore
├── data/
│   └── models/             # Model cache (auto‑created)
└── README.md
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.  
Please ensure the code remains offline‑first and well‑documented.

---

## 📄 License

[MIT](https://choosealicense.com/licenses/mit/) – free for personal and commercial use.

---

## 🙏 Acknowledgements

- [WhisperX](https://github.com/m-bain/whisperX) – core transcription and alignment.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) – optimised runtime.
- [FastAPI](https://fastapi.tiangolo.com) – modern API framework.

---

**Built from scratch – no premade repositories, fully transparent.**  
If you find this project useful, consider starring it ⭐ on GitHub!