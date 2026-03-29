FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY backend/requirements.txt /app/backend/requirements.txt

# Install minimal dependencies
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Create static/data directories for the shared volume mounts
RUN mkdir -p /data/audio /static/webpages /static/cache

# Copy backend app code
COPY backend/app /app/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
