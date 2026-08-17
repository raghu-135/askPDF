FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl g++ libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-control-plane.txt requirements-control-plane.txt
COPY requirements-langgraph-runtime.txt requirements-langgraph-runtime.txt
COPY requirements-runtime.txt requirements-runtime.txt
RUN pip install -r requirements-runtime.txt

COPY app app
COPY runtime_service runtime_service
EXPOSE 8100
CMD ["uvicorn", "runtime_service.main:app", "--host", "0.0.0.0", "--port", "8100"]
