from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from app.runtime.langgraph.checkpointing import delete_agent_checkpoints
from app.services import agent_task_repository as tasks
from app.services.content_store import get_content_store
from app.services.task_artifact_service import cleanup_deleted_task
from app.time_utils import utc_now


logger = logging.getLogger(__name__)
CHECKPOINT_RETENTION_DAYS = 7
MAINTENANCE_INTERVAL_SECONDS = 60.0
MAINTENANCE_BATCH_SIZE = 100
_maintenance_lock = asyncio.Lock()


async def run_task_maintenance(*, batch_size: int = MAINTENANCE_BATCH_SIZE) -> dict[str, int]:
    """Run bounded, idempotent maintenance without overlapping in one process."""
    if _maintenance_lock.locked():
        return {"skipped": 1}
    async with _maintenance_lock:
        bounded = max(1, min(int(batch_size), 500))
        expired_tasks = await tasks.expire_stale_tasks()
        recovered_leases = await tasks.release_stale_task_leases(limit=bounded)
        deleted_tasks = 0
        for task_id in await tasks.list_pending_task_deletions(limit=bounded):
            try:
                await cleanup_deleted_task(task_id)
                deleted_tasks += 1
            except Exception:
                logger.exception("Deep Research task cleanup failed | task_id=%s", task_id)

        store = get_content_store()
        expired_artifacts = 0
        for artifact in await tasks.list_expired_artifacts(limit=bounded):
            await store.delete(artifact.object_key)
            await tasks.mark_artifact_deleted(artifact.task_id, artifact.id)
            expired_artifacts += 1

        live_artifacts = await tasks.list_live_artifacts()
        known_keys = {artifact.object_key for artifact in live_artifacts}
        missing_artifacts = 0
        for artifact in live_artifacts[:bounded]:
            if not await store.exists(artifact.object_key):
                await tasks.mark_artifact_invalid(artifact.task_id, artifact.id, reason="content_missing")
                missing_artifacts += 1

        orphaned_content = 0
        for key in (await store.list_keys("agent-tasks"))[:bounded]:
            if key not in known_keys:
                await store.delete(key)
                orphaned_content += 1

        checkpoint_ids = await tasks.list_terminal_task_checkpoint_ids_before(
            utc_now() - timedelta(days=CHECKPOINT_RETENTION_DAYS),
            limit=bounded,
        )
        deleted_checkpoint_ids = await delete_agent_checkpoints(checkpoint_ids) if checkpoint_ids else []
        await tasks.clear_task_checkpoint_ids(deleted_checkpoint_ids)
        deleted_checkpoints = len(deleted_checkpoint_ids)
        return {
            "expired_tasks": expired_tasks,
            "recovered_leases": recovered_leases,
            "deleted_tasks": deleted_tasks,
            "expired_artifacts": expired_artifacts,
            "missing_artifacts": missing_artifacts,
            "orphaned_content": orphaned_content,
            "deleted_checkpoints": deleted_checkpoints,
        }
