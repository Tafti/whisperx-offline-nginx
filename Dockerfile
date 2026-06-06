FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/data/models \
    TRANSFORMERS_CACHE=/app/data/models/transformers \
    HF_HUB_OFFLINE=0

# =========================
# REMOVE BROKEN NVIDIA APT REPO
# =========================
RUN rm -f /etc/apt/sources.list.d/cuda*.list && \
    rm -f /etc/apt/sources.list.d/nvidia*.list || true

# =========================
# SYSTEM PACKAGES
# =========================
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        libsndfile1 \
        git \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Python symlinks
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

# =========================
# TORCH
# =========================
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 30 \
      --extra-index-url https://download.pytorch.org/whl/cu124 \
      torch torchaudio torchvision

# APP REQUIREMENTS
# =========================
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -vE '^(torch|torchaudio|torchvision)$' requirements.txt > requirements.runtime.txt && \
    pip install --timeout 30 -r requirements.runtime.txt

# Ensure ASGI server binary/module is present at runtime.
RUN python -c "import fastapi, uvicorn"

COPY . .

RUN mkdir -p /app/data/models

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]