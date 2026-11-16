FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY langgraph_runtime/requirements.txt /app/langgraph-runtime-requirements.txt
RUN pip install -r /app/langgraph-runtime-requirements.txt
COPY runtime_protocol /tmp/runtime_protocol
RUN pip install --no-deps /tmp/runtime_protocol

COPY langgraph_runtime /app/langgraph_runtime
COPY scripts/docker_compose_db_migrate.sh scripts/docker_compose_runtime_db_bootstrap.sh \
  scripts/docker_compose_langgraph_start.sh scripts/docker_load_env_secrets.sh /opt/askpdf/
RUN chmod +x /opt/askpdf/docker_compose_db_migrate.sh /opt/askpdf/docker_compose_runtime_db_bootstrap.sh \
  /opt/askpdf/docker_compose_langgraph_start.sh /opt/askpdf/docker_load_env_secrets.sh

EXPOSE 8100
CMD ["uvicorn", "langgraph_runtime.main:app", "--host", "0.0.0.0", "--port", "8100"]
