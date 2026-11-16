FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    SENTENCE_TRANSFORMERS_HOME=/models/hf

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libmagic-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsndfile1 \
    espeak-ng \
    curl \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY rag_service/requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY runtime_protocol /tmp/runtime_protocol
RUN pip install --no-deps /tmp/runtime_protocol

ARG DOWNLOAD_LOCAL_MODELS=true
COPY rag_service/app/services/embedding_tokenizer_registry.py /tmp/embedding_tokenizer_registry.py
COPY rag_service/scripts/download_embedding_tokenizers.py /tmp/download_embedding_tokenizers.py
RUN if [ "$DOWNLOAD_LOCAL_MODELS" = "true" ]; then \
      python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3', trust_remote_code=True); CrossEncoder('BAAI/bge-reranker-v2-m3', trust_remote_code=True); print('Downloaded local embedding + reranker models')"; \
      python /tmp/download_embedding_tokenizers.py; \
    fi

RUN mkdir -p /models/hf /static

COPY rag_service/alembic alembic
COPY rag_service/scripts scripts
COPY rag_service/app app
COPY rag_service/alembic.ini rag_service/main.py ./
COPY scripts/docker_compose_db_migrate.sh scripts/docker_compose_rag_start.sh \
  scripts/docker_load_env_secrets.sh scripts/postgres_advisory_lock.py /opt/askpdf/
RUN chmod +x /opt/askpdf/docker_compose_db_migrate.sh /opt/askpdf/docker_compose_rag_start.sh \
  /opt/askpdf/docker_load_env_secrets.sh

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
