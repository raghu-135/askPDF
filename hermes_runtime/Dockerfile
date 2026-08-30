FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY hermes_runtime/requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY hermes_runtime hermes_runtime

EXPOSE 8200
CMD ["uvicorn", "hermes_runtime.main:app", "--host", "0.0.0.0", "--port", "8200"]
