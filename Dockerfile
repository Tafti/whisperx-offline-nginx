# Use an official Python runtime with CUDA support (optional, for GPU)
# For CPU-only, use: python:3.12-slim
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/data/models \
    TRANSFORMERS_CACHE=/app/data/models/transformers \
    HF_HUB_OFFLINE=0 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=https://package-mirror.liara.ir/repository/pypi/simple

# Remove all existing APT sources (to avoid old mirrors)
RUN rm -rf /etc/apt/sources.list* && \
    rm -rf /etc/apt/sources.list.d/*

# Set ArvanCloud mirrors (detect codename automatically)
RUN set -eux; \
codename=$(grep "VERSION_CODENAME=" /etc/os-release | cut -d= -f2); \
echo "deb http://mirror.arvancloud.ir/debian $codename main" > /etc/apt/sources.list; \
echo "deb http://mirror.arvancloud.ir/debian-security $codename-security main" >> /etc/apt/sources.list; \
cat /etc/apt/sources.list

# Set Liara mirrors (detect codename automatically)
# RUN set -eux; \
#     codename=$(grep "VERSION_CODENAME=" /etc/os-release | cut -d= -f2); \
#     echo "deb http://linux-mirror.liara.ir/debian $codename main" > /etc/apt/sources.list; \
#     echo "deb http://linux-mirror.liara.ir/debian-security $codename-security main" >> /etc/apt/sources.list; \
#     cat /etc/apt/sources.list

# Install system dependencies: FFmpeg and other audio libs
RUN apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for models
RUN mkdir -p /app/data/models

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]