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
    # Chromium dependencies for SingleFile
    chromium \
    chromium-sandbox \
    # Node.js dependencies
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x (required for SingleFile CLI)
RUN curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /usr/share/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY rag_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install SingleFile CLI globally via npm
RUN npm install -g single-file-cli

# Create symlink for chromium to common location
RUN ln -sf /usr/bin/chromium /usr/bin/chromium-browser || true

ARG DOWNLOAD_LOCAL_MODELS=true
RUN if [ "$DOWNLOAD_LOCAL_MODELS" = "true" ]; then \
      python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3', trust_remote_code=True); CrossEncoder('BAAI/bge-reranker-v2-m3', trust_remote_code=True); print('Downloaded local embedding + reranker models')"; \
      python -m spacy download en_core_web_sm; \
    fi

COPY rag_service /app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
