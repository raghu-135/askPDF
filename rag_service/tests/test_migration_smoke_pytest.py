"""Isolated smoke coverage for the complete historical Alembic chain."""

from __future__ import annotations

import os
import subprocess
import asyncio
import importlib.util
from pathlib import Path

import pytest
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


def test_historical_migrations_upgrade_to_irreversible_contract_reset(test_database_url: str):
    """Upgrade through both historical branches into the contract reset head."""

    # Other database fixtures create/drop SQLModel metadata but do not own
    # Alembic-managed functions or its version table. Use a guarded test-only
    # schema reset before treating the shared database as an empty target.
    _reset_test_schema(test_database_url)
    _alembic(test_database_url, "upgrade", "head")
    current = _alembic(test_database_url, "current")
    assert "5c1e8a7d9b3f" in current.stdout

    migration_path = SERVICE_ROOT / "alembic" / "versions" / "5c1e8a7d9b3f_reset_trace_workflow_contract_v1.py"
    spec = importlib.util.spec_from_file_location("contract_reset_migration", migration_path)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    with pytest.raises(RuntimeError, match="intentionally irreversible"):
        migration.downgrade()
