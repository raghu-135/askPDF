"""Explicit preflight for resetting runtime-owned LangGraph checkpoints."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from urllib.parse import urlsplit, urlunsplit

import asyncpg

from app.agent_workflows.repository import AgentWorkflowRepository


async def mark_runs_unresolved(*, limit: int = 500, dry_run: bool = True) -> dict[str, int]:
    """Mark active runs deferred before native checkpoint data is discarded."""
    repository = AgentWorkflowRepository()
    runs = await repository.list_nonterminal_runtime_runs(limit=limit)
    marked = 0
    for run in runs:
        if dry_run:
            continue
        await repository.update_runtime_projection(
            run.id,
            {
                "reconciliation_status": "deferred",
                "binding_status": "legacy_unresolved",
                "checkpoint_reset": {
                    "status": "native_state_discarded",
                    "checkpoint_thread_id": run.checkpoint_thread_id,
                },
            },
        )
        marked += 1
    return {"inspected": len(runs), "marked_unresolved": marked, "dry_run": int(dry_run)}


async def reset_database() -> None:
    """Drop and recreate only the runtime checkpoint database."""
    source = os.environ.get("DATABASE_URL", "")
    parts = urlsplit(source.replace("postgresql+asyncpg://", "postgresql://", 1))
    if not parts.hostname:
        raise RuntimeError("DATABASE_URL is required to reset runtime checkpoints")
    admin_url = urlunsplit((parts.scheme, parts.netloc, "/postgres", "", ""))
    connection = await asyncpg.connect(admin_url)
    try:
        await connection.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'runtime_checkpoints' AND pid <> pg_backend_pid()")
        await connection.execute('DROP DATABASE IF EXISTS "runtime_checkpoints"')
        await connection.execute('CREATE DATABASE "runtime_checkpoints"')
    finally:
        await connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Mark active runs unresolved before resetting runtime checkpoints")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset", action="store_true", help="drop and recreate only runtime_checkpoints")
    parser.add_argument("--confirm-runtime-checkpoint-reset", action="store_true")
    args = parser.parse_args()
    if args.reset and not args.confirm_runtime_checkpoint_reset:
        parser.error("--reset requires --confirm-runtime-checkpoint-reset")

    async def execute() -> dict[str, int]:
        result = await mark_runs_unresolved(limit=args.limit, dry_run=args.dry_run or not args.reset)
        if args.reset:
            await reset_database()
            result["reset"] = 1
        return result

    print(json.dumps(asyncio.run(execute()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
