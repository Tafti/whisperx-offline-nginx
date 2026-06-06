FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/data/models \
    TRANSFORMERS_CACHE=/app/data/models/transformers \
    HF_HUB_OFFLINE=1 \
    PIP_INDEX_URL=https://pypi.org/simple \
    PIP_TRUSTED_HOST=package-mirror.liara.ir

# =========================
# REMOVE BROKEN NVIDIA APT REPO
# =========================
RUN rm -f /etc/apt/sources.list.d/cuda*.list && \
    rm -f /etc/apt/sources.list.d/nvidia*.list || true

# =========================
# ARVANCLOUD MIRROR
# =========================
RUN printf "deb http://archive.ubuntu.com/ubuntu jammy main restricted universe multiverse\n\
deb http://archive.ubuntu.com/ubuntu jammy-updates main restricted universe multiverse\n\
deb http://archive.ubuntu.com/ubuntu jammy-backports main restricted universe multiverse\n\
deb http://security.ubuntu.com/ubuntu jammy-security main restricted universe multiverse\n" \
> /etc/apt/sources.list\
printf "deb http://mirror.arvancloud.ir/ubuntu jammy main restricted universe multiverse\n\
deb http://mirror.arvancloud.ir/ubuntu jammy-updates main restricted universe multiverse\n\
deb http://mirror.arvancloud.ir/ubuntu jammy-backports main restricted universe multiverse\n\
deb http://mirror.arvancloud.ir/ubuntu jammy-security main restricted universe multiverse\n" \
> /etc/apt/sources.list

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
RUN for i in $(seq 1 10); do \
      pip install --timeout 300 \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        torch torchaudio torchvision && break; \
      sleep 15; \
    done

# =========================
# APP REQUIREMENTS
# =========================
RUN grep -vE '^(torch|torchaudio|torchvision)$' requirements.txt > requirements.runtime.txt && \
    pip install --timeout 180 -r requirements.runtime.txt

COPY . .

RUN mkdir -p /app/data/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]