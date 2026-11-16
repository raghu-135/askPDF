"""Acquire a Postgres advisory lock before running one-shot schema migrations."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from urllib.parse import urlparse

import psycopg


def _admin_database_url(database_url: str) -> str:
    raw = database_url.replace("postgresql+asyncpg://", "postgresql://")
    parsed = urlparse(raw)
    if not parsed.hostname:
        raise RuntimeError(f"Invalid DATABASE_URL host: {database_url!r}")
    user = parsed.username or "postgres"
    password = parsed.password or "postgres"
    port = parsed.port or 5432
    return f"postgresql://{user}:{password}@{parsed.hostname}:{port}/postgres"


@contextmanager
def advisory_lock(lock_key: str, *, database_url: str | None = None, wait_seconds: int = 600):
    url = (
        database_url
        or os.environ.get("DATABASE_URL")
        or os.environ.get("AGENT_RUNTIME_EXECUTION_DATABASE_URL")
        or os.environ.get("AGENT_CHECKPOINT_DATABASE_URL")
    )
    if not url:
        raise RuntimeError("DATABASE_URL is required for advisory lock")

    admin_url = _admin_database_url(url)
    with psycopg.connect(admin_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (lock_key,))
            try:
                yield
            finally:
                cur.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (lock_key,))


def main() -> None:
    lock_key = sys.argv[1] if len(sys.argv) > 1 else "askpdf-migrate"
    module = sys.argv[2]
    import subprocess

    with advisory_lock(lock_key):
        subprocess.check_call([sys.executable, "-m", module])


if __name__ == "__main__":
    main()