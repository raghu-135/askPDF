"""Apply the dedicated Alembic migrations to the runtime database."""

import asyncio
import os
import subprocess
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

RUNTIME_DB_MIGRATION_LOCK_KEY = "askpdf-runtime-db-migrate"


RUNTIME_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = RUNTIME_ROOT.parent


def _run_alembic(*args: str) -> None:
    subprocess.run(["alembic", *args], check=True, cwd=REPOSITORY_ROOT)


async def _run_migrations_under_lock(async_database_url: str) -> None:
    engine = create_async_engine(async_database_url, poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            await connection.execute(
                text("SELECT pg_advisory_lock(hashtextextended(:key, 0))"),
                {"key": RUNTIME_DB_MIGRATION_LOCK_KEY},
            )
            try:
                _run_alembic("-c", str(RUNTIME_ROOT / "alembic.ini"), "upgrade", "head")
            finally:
                await connection.execute(
                    text("SELECT pg_advisory_unlock(hashtextextended(:key, 0))"),
                    {"key": RUNTIME_DB_MIGRATION_LOCK_KEY},
                )
    finally:
        await engine.dispose()


def main() -> None:
    if os.environ.get("RUN_RUNTIME_DB_MIGRATIONS", "true").strip().lower() in {"0", "false", "no", "off"}:
        print("Runtime database migrations are disabled because RUN_RUNTIME_DB_MIGRATIONS=false.", flush=True)
        return

    database_url = os.environ.get("AGENT_RUNTIME_EXECUTION_DATABASE_URL")
    if not database_url:
        raise RuntimeError("AGENT_RUNTIME_EXECUTION_DATABASE_URL is required")

    async_database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    os.environ["DATABASE_URL"] = async_database_url
    asyncio.run(_run_migrations_under_lock(async_database_url))


if __name__ == "__main__":
    main()
