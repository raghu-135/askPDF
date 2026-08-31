FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY langgraph_runtime/requirements.txt /app/langgraph-runtime-requirements.txt
RUN pip install -r /app/langgraph-runtime-requirements.txt

COPY langgraph_runtime /app/langgraph_runtime
COPY rag_service/runtime_protocol /app/runtime_protocol

EXPOSE 8100
CMD ["uvicorn", "langgraph_runtime.main:app", "--host", "0.0.0.0", "--port", "8100"]
