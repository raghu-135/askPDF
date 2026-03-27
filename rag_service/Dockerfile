FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    SENTENCE_TRANSFORMERS_HOME=/models/hf \
    DEFAULT_EMBEDDING_MODEL=BAAI/bge-m3 \
    DEFAULT_RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
    LOCAL_EMBEDDING_MODELS=BAAI/bge-m3 \
    USE_LOCAL_EMBEDDINGS=true \
    USE_LOCAL_RERANKER=true \
    EMBEDDING_DEVICE=cpu \
    RERANKER_DEVICE=cpu \
    RERANK_TOP_K_MULTIPLIER=4

WORKDIR /app

RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libmagic-dev \
    libgl1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY rag_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

ARG DOWNLOAD_LOCAL_MODELS=true
RUN if [ "$DOWNLOAD_LOCAL_MODELS" = "true" ]; then \
      python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3', trust_remote_code=True); CrossEncoder('BAAI/bge-reranker-v2-m3', trust_remote_code=True); print('Downloaded local embedding + reranker models')"; \
    fi

COPY rag_service /app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
