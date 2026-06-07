FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Remove stale NVIDIA apt sources
RUN rm -f /etc/apt/sources.list.d/cuda*.list && \
    rm -f /etc/apt/sources.list.d/nvidia*.list || true

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        ffmpeg libsndfile1 git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Build dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --timeout 120 \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        torch torchaudio torchvision

# Install app dependencies excluding torch packages
RUN set -eux; \
    grep -Ev '^(torch|torchaudio|torchvision)$' requirements.txt > /tmp/requirements.runtime.txt; \
    pip install --timeout 120 -r /tmp/requirements.runtime.txt; \
    rm -f /tmp/requirements.runtime.txt

COPY main.py config.py ./
COPY utils ./utils

# Runtime stage
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/data/models \
    HF_HUB_CACHE=/app/data/models/hub \
    TRANSFORMERS_CACHE=/app/data/models/transformers

RUN rm -f /etc/apt/sources.list.d/cuda*.list && \
    rm -f /etc/apt/sources.list.d/nvidia*.list || true

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg libsndfile1 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

RUN mkdir -p /app/data/models

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
