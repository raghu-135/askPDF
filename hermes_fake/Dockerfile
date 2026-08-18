FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn httpx
COPY hermes_fake hermes_fake

EXPOSE 8000
CMD ["uvicorn", "hermes_fake.main:app", "--host", "0.0.0.0", "--port", "8000"]
