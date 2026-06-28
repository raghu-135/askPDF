"""
Migration bootstrap for container startup.

This keeps Alembic as the schema authority while supporting databases that were
created before Alembic was wired into docker compose.
"""

import asyncio
import os
import subprocess
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool


INITIAL_REVISION = "8c8d6eac150a"
HEAD_REVISION = "head"
BASELINE_TABLES = {
    "files",
    "messages",
    "thread_file_annotations",
    "thread_files",
    "thread_stats",
    "threads",
}


def _env_flag_enabled(name: str, default: bool = True) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def _run_alembic(*args: str) -> None:
    subprocess.run(["alembic", *args], check=True)


async def _bootstrap_legacy_database() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable is required")

    engine = create_async_engine(database_url, poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            alembic_version_exists = await connection.scalar(
                text(
                    """
                    select exists (
                        select 1
                        from information_schema.tables
                        where table_schema = 'public'
                          and table_name = 'alembic_version'
                    )
                    """
                )
            )
            if alembic_version_exists:
                return

            rows = await connection.execute(
                text(
                    """
                    select table_name
                    from information_schema.tables
                    where table_schema = 'public'
                    """
                )
            )
            existing_tables = {row[0] for row in rows}
            if not existing_tables:
                return

            missing_tables = BASELINE_TABLES - existing_tables
            if missing_tables:
                missing = ", ".join(sorted(missing_tables))
                raise RuntimeError(
                    "Database has application tables but is not a complete "
                    f"legacy baseline. Missing tables: {missing}"
                )

            thread_metadata_exists = await connection.scalar(
                text(
                    """
                    select exists (
                        select 1
                        from information_schema.columns
                        where table_schema = 'public'
                          and table_name = 'threads'
                          and column_name = 'thread_metadata'
                    )
                    """
                )
            )
    finally:
        await engine.dispose()

    if thread_metadata_exists:
        _run_alembic("stamp", HEAD_REVISION)
    else:
        _run_alembic("stamp", INITIAL_REVISION)


def main() -> None:
    if not _env_flag_enabled("RUN_DB_MIGRATIONS", default=True):
        print(
            "Database migrations are disabled because RUN_DB_MIGRATIONS=false. "
            "Starting without applying schema changes.",
            flush=True,
        )
        return

    try:
        asyncio.run(_bootstrap_legacy_database())
        _run_alembic("upgrade", HEAD_REVISION)
    except Exception as exc:
        print("", file=sys.stderr, flush=True)
        print("Database migration failed.", file=sys.stderr, flush=True)
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        print(
            "To start the app without running migrations, set "
            "RUN_DB_MIGRATIONS=false and run docker compose up again. "
            "Only do this if the database schema is already compatible with "
            "the application version you are starting.",
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
