"""Isolated smoke coverage for the complete historical Alembic chain."""

from __future__ import annotations

import os
import subprocess
import asyncio
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine


SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _alembic(test_database_url: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["DATABASE_URL"] = test_database_url
    result = subprocess.run(
        ["alembic", *arguments],
        cwd=SERVICE_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"alembic {' '.join(arguments)} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
        )
    return result


def _reset_test_schema(test_database_url: str) -> None:
    database = make_url(test_database_url).database or ""
    if not database.startswith("test_"):
        raise RuntimeError(f"Refusing to reset non-test database: {database}")

    async def reset() -> None:
        engine = create_async_engine(test_database_url)
        try:
            async with engine.begin() as connection:
                await connection.execute(text("drop schema public cascade"))
                await connection.execute(text("create schema public"))
        finally:
            await engine.dispose()

    asyncio.run(reset())


def test_historical_migrations_upgrade_and_latest_downgrade(test_database_url: str):
    """Upgrade an empty database through history and smoke-test the head downgrade."""

    # Other database fixtures create/drop SQLModel metadata but do not own
    # Alembic-managed functions or its version table. Use a guarded test-only
    # schema reset before treating the shared database as an empty target.
    _reset_test_schema(test_database_url)
    _alembic(test_database_url, "upgrade", "k5e2a8c4d7f1")
    current = _alembic(test_database_url, "current")
    assert "k5e2a8c4d7f1" in current.stdout

    try:
        _alembic(test_database_url, "downgrade", "-1")
        downgraded = _alembic(test_database_url, "current")
        assert "j4d9e6f1b3c5" in downgraded.stdout
    finally:
        _alembic(test_database_url, "upgrade", "k5e2a8c4d7f1")
