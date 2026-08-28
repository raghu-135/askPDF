"""Isolated smoke coverage for the complete historical Alembic chain."""

from __future__ import annotations

import os
import subprocess
import asyncio
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
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


def _runtime_alembic(test_database_url: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["DATABASE_URL"] = test_database_url
    result = subprocess.run(
        ["alembic", "-c", "runtime_alembic.ini", *arguments],
        cwd=SERVICE_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"runtime alembic {' '.join(arguments)} failed ({result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
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


def test_application_migrations_upgrade_without_resetting_data(test_database_url: str):
    """Upgrade the existing branch history without deleting application data."""

    # Other database fixtures create/drop SQLModel metadata but do not own
    # Alembic-managed functions or its version table. Use a guarded test-only
    # schema reset before treating the shared database as an empty target.
    _reset_test_schema(test_database_url)
    _alembic(test_database_url, "upgrade", "a8d3f1c6e4b2")

    async def seed_existing_data() -> None:
        engine = create_async_engine(test_database_url)
        try:
            async with engine.begin() as connection:
                await connection.execute(
                    text(
                        """
                        insert into projects (id, name, description, embedding_model, settings_json)
                        values ('migration-project', 'Migration test project', '', 'test-model', '{}'::jsonb)
                        """
                    )
                )
                await connection.execute(
                    text(
                        "insert into threads (id, project_id, name, embedding_model) "
                        "values ('migration-thread', 'migration-project', 'Migration test', 'test-model')"
                    )
                )
                await connection.execute(
                    text(
                        """
                        insert into agent_workflows (id, name, description, visibility, is_builtin, spec_json, validation_result_json, metadata_json)
                        values ('migration-workflow', 'Migration test workflow', '', 'builtin', false, '{}'::jsonb, '{}'::jsonb, '{}'::jsonb)
                        """
                    )
                )
                await connection.execute(
                    text(
                        """
                        insert into agent_tasks (id, thread_id, workflow_id, objective, objective_hash, create_idempotency_key)
                        values ('migration-task', 'migration-thread', 'migration-workflow', 'Keep this task', 'migration-task-hash', 'migration-task-key')
                        """
                    )
                )
                await connection.execute(
                    text(
                        """
                        insert into agent_runs (id, thread_id, workflow_id, debug_trace_json)
                        values ('migration-run', 'migration-thread', 'migration-workflow', '{"trace":"keep"}'::jsonb)
                        """
                    )
                )
                await connection.execute(
                    text(
                        """
                        insert into agent_task_events (id, task_id, sequence, event_type, actor_type, payload_json)
                        values ('migration-event', 'migration-task', 1, 'run.started', 'system', jsonb_build_object('keep', true))
                        """
                    )
                )
        finally:
            await engine.dispose()

    asyncio.run(seed_existing_data())
    _alembic(test_database_url, "upgrade", "head")
    current = _alembic(test_database_url, "current")
    assert "a9c7e1f3b5d2" in current.stdout

    async def verify_existing_data() -> tuple[int, int, int, int, dict]:
        engine = create_async_engine(test_database_url)
        try:
            async with engine.connect() as connection:
                counts = await connection.execute(
                    text(
                        """
                        select
                          (select count(*) from agent_workflows where id = 'migration-workflow'),
                          (select count(*) from agent_tasks where id = 'migration-task'),
                          (select count(*) from agent_runs where id = 'migration-run'),
                          (select count(*) from agent_task_events where id = 'migration-event'),
                          (select debug_trace_json from agent_runs where id = 'migration-run')
                        """
                    )
                )
                row = counts.one()
                return row[0], row[1], row[2], row[3], row[4]
        finally:
            await engine.dispose()

    assert asyncio.run(verify_existing_data()) == (1, 1, 1, 1, {"trace": "keep"})


def test_runtime_migrations_have_an_independent_single_head():
    config = Config(str(SERVICE_ROOT / "runtime_alembic.ini"))
    config.set_main_option("script_location", str(SERVICE_ROOT / "runtime_alembic"))
    scripts = ScriptDirectory.from_config(config)
    assert set(scripts.get_heads()) == {"r1_runtime_schema"}


def test_fresh_runtime_database_reaches_complete_schema(test_database_url: str):
    _runtime_alembic(test_database_url, "upgrade", "head")

    async def inspect_schema() -> set[str]:
        engine = create_async_engine(test_database_url)
        try:
            async with engine.connect() as connection:
                result = await connection.execute(
                    text(
                        """
                        select table_name
                          from information_schema.tables
                         where table_schema = 'public'
                           and table_name in ('runtime_executions', 'runtime_events', 'runtime_operations')
                        """
                    )
                )
                return {row[0] for row in result}
        finally:
            await engine.dispose()

    assert asyncio.run(inspect_schema()) == {
        "runtime_executions",
        "runtime_events",
        "runtime_operations",
    }
