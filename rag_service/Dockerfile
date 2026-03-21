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
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY rag_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

ARG DOWNLOAD_LOCAL_MODELS=true
RUN if [ "$DOWNLOAD_LOCAL_MODELS" = "true" ]; then \
    python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
    from fastembed.rerank.cross_encoder import TextCrossEncoder; \
    TextEmbedding('mixedbread-ai/mxbai-embed-large-v1'); \
    SparseTextEmbedding('Qdrant/bm25'); \
    TextCrossEncoder('jinaai/jina-reranker-v1-tiny-en'); \
    print('Downloaded FastEmbed models: MXBAI-Large, BM25 Sparse, and Jina-Reranker-Tiny-En')"; \
    fi

COPY rag_service /app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
