#!/bin/sh
# One-shot migration job for Compose: wait for DB DNS/TCP, then run migrations with retries.
set -eu

MODULE="${1:?python module e.g. app.db.migrate}"
DB_URL_ENV="${2:-DATABASE_URL}"

if ! python -c "import os, sys; sys.exit(0 if os.environ.get('$DB_URL_ENV') else 1)"; then
  echo "Missing required environment variable: $DB_URL_ENV" >&2
  exit 1
fi

python - "$DB_URL_ENV" <<'PY'
import os
import socket
import sys
import time
from urllib.parse import urlparse

env_name = sys.argv[1]
raw = os.environ[env_name].replace("postgresql+asyncpg://", "postgresql://")
parsed = urlparse(raw)
host = parsed.hostname or "postgresql"
port = parsed.port or 5432

for attempt in range(60):
    try:
        socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        with socket.create_connection((host, port), timeout=3):
            break
    except OSError:
        time.sleep(1)
else:
    sys.exit(f"database endpoint {host}:{port} not reachable after 60s")
PY

attempt=1
while [ "$attempt" -le 15 ]; do
  if python -m "$MODULE"; then
    exit 0
  fi
  echo "Migration attempt $attempt failed; retrying..." >&2
  attempt=$((attempt + 1))
  sleep 2
done

exit 1