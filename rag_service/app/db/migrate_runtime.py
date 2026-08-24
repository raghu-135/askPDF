"""Apply Alembic migrations to the runtime-owned checkpoint database."""

import asyncio
import os
import subprocess

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool


BASELINE_REVISION = "9b4d6e2f1a7c"
RUNTIME_HEAD_REVISION = "1e8f3a7c5b2d"


def _async_database_url(database_url: str) -> str:
    """Use the asyncpg driver required by the application's Alembic env."""
    if database_url.startswith("postgresql+asyncpg://"):
        return database_url
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return database_url


async def _has_alembic_version(database_url: str) -> bool:
    engine = create_async_engine(_async_database_url(database_url), poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            table_exists = await connection.scalar(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = 'alembic_version'
                    )
                    """
                )
            )
            if not table_exists:
                return False
            return bool(await connection.scalar(text("SELECT EXISTS (SELECT 1 FROM alembic_version)")))
    finally:
        await engine.dispose()


def _run_alembic(*args: str) -> None:
    subprocess.run(["alembic", *args], check=True)


def main() -> None:
    if os.environ.get("RUN_RUNTIME_DB_MIGRATIONS", "true").strip().lower() in {"0", "false", "no", "off"}:
        print("Runtime database migrations are disabled because RUN_RUNTIME_DB_MIGRATIONS=false.", flush=True)
        return

    database_url = os.environ.get("AGENT_RUNTIME_EXECUTION_DATABASE_URL")
    if not database_url:
        raise RuntimeError("AGENT_RUNTIME_EXECUTION_DATABASE_URL is required")

    # Reuse the application's standard Alembic DATABASE_URL convention.
    os.environ["DATABASE_URL"] = _async_database_url(database_url)
    os.environ["ALEMBIC_RUNTIME_DATABASE"] = "true"

    if not asyncio.run(_has_alembic_version(database_url)):
        # The runtime schema historically had no Alembic version table. Stamp
        # the last application revision so only runtime-owned revisions run.
        _run_alembic("stamp", BASELINE_REVISION)
    # Upgrade only the runtime branch. The application database has other
    # migration heads that are not valid against the checkpoint database.
    _run_alembic("upgrade", RUNTIME_HEAD_REVISION)


if __name__ == "__main__":
    main()
