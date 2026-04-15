FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    SENTENCE_TRANSFORMERS_HOME=/models/hf

WORKDIR /app

RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libmagic-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsndfile1 \
    espeak-ng \
    curl \
    git \
    wget \
    gnupg \
    ca-certificates \
    # Playwright/Chromium dependencies
    libdbus-1-3 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libgtk-3-0 \
    # Chrome dependencies for nodriver
    fonts-liberation \
    libappindicator3-1 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdrm2 \
    libgbm1 \
    libnspr4 \
    libnss3 \
    libxss1 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Chromium for nodriver (supports both AMD64 and ARM64)
RUN apt-get update \
    && apt-get install -y chromium \
    && rm -rf /var/lib/apt/lists/*

COPY rag_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install Playwright Chromium browser
RUN playwright install chromium

ARG DOWNLOAD_LOCAL_MODELS=true
RUN if [ "$DOWNLOAD_LOCAL_MODELS" = "true" ]; then \
      python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3', trust_remote_code=True); CrossEncoder('BAAI/bge-reranker-v2-m3', trust_remote_code=True); print('Downloaded local embedding + reranker models')"; \
      python -m spacy download en_core_web_sm; \
    fi

COPY rag_service /app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
