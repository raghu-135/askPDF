"""Apply Alembic migrations to the runtime-owned checkpoint database."""

import asyncio
import os
import subprocess

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool


BASELINE_REVISION = "9b4d6e2f1a7c"


async def _has_alembic_version(database_url: str) -> bool:
    engine = create_async_engine(database_url, poolclass=NullPool)
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
    database_url = os.environ.get("AGENT_RUNTIME_EXECUTION_DATABASE_URL")
    if not database_url:
        raise RuntimeError("AGENT_RUNTIME_EXECUTION_DATABASE_URL is required")

    # Reuse the application's standard Alembic DATABASE_URL convention.
    os.environ["DATABASE_URL"] = database_url
    os.environ["ALEMBIC_RUNTIME_DATABASE"] = "true"

    if not asyncio.run(_has_alembic_version(database_url)):
        # The runtime schema historically had no Alembic version table. Stamp
        # the last application revision so only runtime-owned revisions run.
        _run_alembic("stamp", BASELINE_REVISION)
    _run_alembic("upgrade", "head")


if __name__ == "__main__":
    main()
