# Manga Translator — CPU-only local / self-hosted
# Build:  docker build -t manga-translator .
# Run:    docker compose up  (recommended)  or  see docker-compose.yml

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps for OpenCV, Paddle, Playwright, fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
        fonts-liberation \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# Python deps first (better layer cache)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Playwright Chromium (needed for JS-rendered chapter sites)
RUN playwright install --with-deps chromium

# Application code
COPY app/ ./app/
COPY run.py convert_model.py ./

# Create runtime dirs (will be overridden by volumes)
RUN mkdir -p data/raw data/processed data/output models logs \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["python", "run.py"]
