"""Isolated smoke coverage for the complete historical Alembic chain."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _alembic(test_database_url: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["DATABASE_URL"] = test_database_url
    return subprocess.run(
        ["alembic", *arguments],
        cwd=SERVICE_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_historical_migrations_upgrade_and_latest_downgrade(test_database_url: str):
    """Upgrade an empty database through history and smoke-test the head downgrade."""

    _alembic(test_database_url, "upgrade", "head")
    current = _alembic(test_database_url, "current")
    assert "7f3c1a9d5e2b" in current.stdout

    try:
        _alembic(test_database_url, "downgrade", "-1")
        downgraded = _alembic(test_database_url, "current")
        assert "d1e8f4a9b2c7" in downgraded.stdout
    finally:
        _alembic(test_database_url, "upgrade", "head")

