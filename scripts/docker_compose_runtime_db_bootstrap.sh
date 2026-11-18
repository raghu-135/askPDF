#!/bin/sh
# One-shot: ensure runtime_checkpoints DB exists, then apply LangGraph runtime migrations.
set -eu

python - <<'PY'
import os
import socket
import time

import psycopg

host = os.environ.get("RUNTIME_BOOTSTRAP_PG_HOST", "postgresql")
user = os.environ.get("RUNTIME_BOOTSTRAP_PG_USER", "postgres")
password = os.environ.get("RUNTIME_BOOTSTRAP_PG_PASSWORD", "postgres")
port = int(os.environ.get("RUNTIME_BOOTSTRAP_PG_PORT", "5432"))

for _ in range(60):
    try:
        socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        with socket.create_connection((host, port), timeout=3):
            break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit(f"database endpoint {host}:{port} not reachable after 60s")

with psycopg.connect(
    host=host, port=port, user=user, password=password, dbname="askpdf", autocommit=True
) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname='runtime_checkpoints'")
        if cur.fetchone() is None:
            cur.execute("CREATE DATABASE runtime_checkpoints")
            print("Created database runtime_checkpoints", flush=True)
PY

exec /bin/sh /opt/askpdf/docker_compose_db_migrate.sh langgraph_runtime.migrate