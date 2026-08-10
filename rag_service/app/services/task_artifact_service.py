from __future__ import annotations

import hashlib
import uuid
from datetime import timedelta
from typing import Optional

from app.db.models_sqlmodel import AgentTaskArtifact
from app.services.agent_task_repository import (
    list_artifacts_for_threads,
    list_task_checkpoint_ids_for_threads,
    mark_artifact_deleted,
    register_artifact,
)
from app.services.content_store import get_content_store, task_artifact_content_key
from app.time_utils import utc_now


MAX_SINGLE_ARTIFACT_BYTES = 10_485_760
ARTIFACT_RETENTION_DAYS = 30


def artifact_ownership_key(*, todo_id: Optional[str], subagent_run_id: Optional[str]) -> str:
    if subagent_run_id:
        return f"subagent:{subagent_run_id}"
    if todo_id:
        return f"todo:{todo_id}"
    return "run"


async def delete_task_resources_for_threads(thread_ids: list[str]) -> None:
    """Delete task content and checkpoints before relational ownership rows cascade."""
    from app.agent_workflows.checkpointing import delete_agent_checkpoints

    artifacts = await list_artifacts_for_threads(thread_ids)
    store = get_content_store()
    for artifact in artifacts:
        await store.delete(artifact.object_key)
        await mark_artifact_deleted(artifact.task_id, artifact.id)
    checkpoint_ids = await list_task_checkpoint_ids_for_threads(thread_ids)
    if checkpoint_ids:
        await delete_agent_checkpoints(checkpoint_ids)


async def cleanup_deleted_task(task_id: str) -> None:
    """Idempotently remove content/checkpoints for one hidden terminal task."""
    from app.agent_workflows.checkpointing import delete_agent_checkpoints
    from app.services import agent_task_repository as tasks

    store = get_content_store()
    for artifact in await tasks.list_artifacts(task_id):
        await store.delete(artifact.object_key)
        await tasks.mark_artifact_deleted(task_id, artifact.id)
    runs = await tasks.list_task_runs(task_id)
    checkpoint_ids = [str(run.checkpoint_thread_id or run.id) for run in runs]
    if checkpoint_ids:
        await delete_agent_checkpoints(checkpoint_ids)
    await tasks.mark_task_deletion_completed(task_id)


async def persist_task_artifact(
    *,
    task_id: str,
    agent_run_id: str,
    kind: str,
    content: bytes | str,
    media_type: str = "text/markdown",
    todo_id: Optional[str] = None,
    subagent_run_id: Optional[str] = None,
    provenance: Optional[dict] = None,
    source_refs: Optional[dict] = None,
    sensitivity: str = "private",
    supersedes_id: Optional[str] = None,
) -> AgentTaskArtifact:
    body = content.encode("utf-8") if isinstance(content, str) else bytes(content)
    if len(body) > MAX_SINGLE_ARTIFACT_BYTES:
        raise ValueError("artifact exceeds the 10 MB per-object limit")
    artifact_id = str(uuid.uuid4())
    object_key = task_artifact_content_key(task_id, agent_run_id, artifact_id)
    digest = hashlib.sha256(body).hexdigest()
    store = get_content_store()
    await store.put(object_key, body, expected_sha256=digest)
    metadata = AgentTaskArtifact(
        id=artifact_id,
        task_id=task_id,
        agent_run_id=agent_run_id,
        todo_id=todo_id,
        subagent_run_id=subagent_run_id,
        ownership_key=artifact_ownership_key(todo_id=todo_id, subagent_run_id=subagent_run_id),
        kind=kind,
        object_key=object_key,
        media_type=media_type,
        byte_size=len(body),
        sha256=digest,
        provenance_json=provenance or {},
        source_refs_json=source_refs or {},
        sensitivity=sensitivity,
        supersedes_id=supersedes_id,
        retention_until=utc_now() + timedelta(days=ARTIFACT_RETENTION_DAYS),
    )
    try:
        persisted, duplicate = await register_artifact(metadata)
    except Exception:
        await store.delete(object_key)
        raise
    if duplicate:
        await store.delete(object_key)
    return persisted
