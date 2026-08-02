from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.future import select

from app.db.models_sqlmodel import (
    AgentRun,
    AgentWorkflow,
    ChatTurn,
    File,
    Memory,
    MemoryOverride,
    Project,
    ProjectFile,
    Thread,
    ThreadDocumentAnnotation,
    ThreadFile,
)
from app.services import project_lifecycle_service
from app.time_utils import utc_now


@pytest.fixture
def lifecycle_sessionmaker(engine, monkeypatch):
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(project_lifecycle_service, "async_session_maker", maker)
    monkeypatch.setattr(
        project_lifecycle_service,
        "_default_project_id",
        AsyncMock(return_value="default-project"),
    )
    monkeypatch.setattr(
        project_lifecycle_service,
        "require_embedding_model_ready",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        project_lifecycle_service,
        "index_memory_record",
        AsyncMock(return_value=1),
    )
    return maker


def _memory(memory_id, scope_type, scope_id, content):
    return Memory(
        id=memory_id,
        scope_type=scope_type,
        scope_id=scope_id,
        content=content,
        embedding_model="BAAI/bge-m3",
        content_hash=f"hash-{memory_id}",
        created_at=utc_now(),
    )


@pytest.mark.asyncio
async def test_clone_project_without_threads_copies_knowledge_and_project_memory(
    lifecycle_sessionmaker,
):
    async with lifecycle_sessionmaker() as session:
        async with session.begin():
            session.add(Project(
                id="source-project",
                name="Source",
                description="Description",
                embedding_model="BAAI/bge-m3",
                settings_json={"memory": {"project_reads_user_memory": True}},
            ))
            session.add(File(file_hash="shared-file", file_name="paper.pdf", source_type="pdf"))
            await session.flush()
            session.add(ProjectFile(project_id="source-project", file_hash="shared-file"))
            session.add(Thread(
                id="source-thread",
                project_id="source-project",
                name="Conversation",
                embedding_model="BAAI/bge-m3",
            ))
            await session.flush()
            session.add(_memory(
                "source-project-memory",
                "project",
                "source-project",
                "Project fact",
            ))
            session.add(_memory(
                "source-thread-memory",
                "thread",
                "source-thread",
                "Thread fact",
            ))
            session.add(_memory(
                "global-memory",
                "user",
                "default",
                "Global fact",
            ))
            await session.flush()
            session.add(MemoryOverride(
                overriding_memory_id="source-project-memory",
                overridden_memory_id="global-memory",
            ))

    result = await project_lifecycle_service.clone_project(
        "source-project",
        name="Source Copy",
        include_threads=False,
    )

    assert result["counts"]["project_files"] == 1
    assert result["counts"]["project_memories"] == 1
    assert result["counts"]["threads"] == 0
    async with lifecycle_sessionmaker() as session:
        cloned = await session.get(Project, result["project"].id)
        cloned_threads = list((await session.execute(
            select(Thread).where(Thread.project_id == cloned.id)
        )).scalars().all())
        cloned_links = list((await session.execute(
            select(ProjectFile).where(ProjectFile.project_id == cloned.id)
        )).scalars().all())
        cloned_memories = list((await session.execute(
            select(Memory).where(Memory.scope_type == "project", Memory.scope_id == cloned.id)
        )).scalars().all())
        cloned_override = (await session.execute(
            select(MemoryOverride).where(
                MemoryOverride.overriding_memory_id == cloned_memories[0].id
            )
        )).scalar_one()

    assert cloned.name == "Source Copy"
    assert cloned.embedding_model == "BAAI/bge-m3"
    assert cloned_threads == []
    assert [row.file_hash for row in cloned_links] == ["shared-file"]
    assert len(cloned_memories) == 1
    assert cloned_memories[0].id != "source-project-memory"
    assert cloned_memories[0].source_refs_json["fork_origin"]["source_memory_id"] == "source-project-memory"
    assert cloned_override.overridden_memory_id == "global-memory"
    assert result["counts"]["memory_overrides"] == 1


@pytest.mark.asyncio
async def test_clone_with_threads_copies_completed_history_annotations_and_trace(
    lifecycle_sessionmaker,
):
    now = utc_now()
    debug = {
        "version": 1,
        "trace": {"run_id": "source-run", "thread_id": "source-thread", "spans": []},
        "summary": {"status": "completed"},
    }
    async with lifecycle_sessionmaker() as session:
        async with session.begin():
            session.add(Project(
                id="source-project",
                name="Source",
                embedding_model="BAAI/bge-m3",
            ))
            session.add(File(file_hash="file-1", file_name="paper.pdf", source_type="pdf"))
            await session.flush()
            session.add(ProjectFile(project_id="source-project", file_hash="file-1"))
            session.add(Thread(
                id="source-thread",
                project_id="source-project",
                name="Conversation",
                embedding_model="BAAI/bge-m3",
                settings={"memory": {"thread_reads_project_memory": True}},
                thread_metadata={
                    "memory_curator": {
                        "reviewed_through_turn_id": "source-turn",
                        "reviewed_through_created_at": now.isoformat(),
                    }
                },
            ))
            await session.flush()
            session.add(ThreadFile(thread_id="source-thread", file_hash="file-1"))
            session.add(ThreadDocumentAnnotation(
                thread_id="source-thread",
                file_hash="file-1",
                annotations=[{"id": "note-1"}],
            ))
            session.add(AgentWorkflow(
                id="workflow-1",
                name="Workflow 1",
                spec_json={},
                validation_result_json={},
            ))
            await session.flush()
            session.add(AgentRun(
                id="source-run",
                thread_id="source-thread",
                workflow_id="workflow-1",
                status="completed",
                checkpoint_thread_id="checkpoint-source",
                pending_interrupt_json={"id": "interrupt"},
                debug_trace_json=debug,
                completed_at=now,
            ))
            await session.flush()
            session.add(ChatTurn(
                id="source-turn",
                thread_id="source-thread",
                agent_run_id="source-run",
                agent_run_turn_kind="answer",
                agent_run_sequence=1,
                agent_trace_refs_json={"span_ids": ["span-1"]},
                status="completed",
                payload={"question": "Q", "answer": "A"},
                completed_at=now,
            ))
            session.add(ChatTurn(
                id="failed-turn",
                thread_id="source-thread",
                status="failed",
                payload={"question": "Failed", "answer": None},
            ))
            session.add(_memory(
                "source-thread-memory",
                "thread",
                "source-thread",
                "Thread fact",
            ))

    result = await project_lifecycle_service.clone_project(
        "source-project",
        name="Full Copy",
        include_threads=True,
    )

    assert result["counts"]["threads"] == 1
    assert result["counts"]["turns"] == 1
    assert result["counts"]["agent_runs"] == 1
    assert result["counts"]["annotations"] == 1
    async with lifecycle_sessionmaker() as session:
        cloned_thread = (await session.execute(
            select(Thread).where(Thread.project_id == result["project"].id)
        )).scalar_one()
        cloned_turn = (await session.execute(
            select(ChatTurn).where(ChatTurn.thread_id == cloned_thread.id)
        )).scalar_one()
        cloned_run = await session.get(AgentRun, cloned_turn.agent_run_id)
        cloned_annotation = (await session.execute(
            select(ThreadDocumentAnnotation).where(
                ThreadDocumentAnnotation.thread_id == cloned_thread.id
            )
        )).scalar_one()

    assert cloned_thread.thread_metadata["project_clone"]["source_thread_id"] == "source-thread"
    assert cloned_thread.thread_metadata["memory_curator"]["reviewed_through_turn_id"] == cloned_turn.id
    assert cloned_turn.id != "source-turn"
    assert cloned_run.id != "source-run"
    assert cloned_run.thread_id == cloned_thread.id
    assert cloned_run.checkpoint_thread_id is None
    assert cloned_run.pending_interrupt_json is None
    assert cloned_run.run_metadata_json["historical_clone"] is True
    assert cloned_run.debug_trace_json["trace"]["run_id"] == cloned_run.id
    assert cloned_run.debug_trace_json["trace"]["thread_id"] == cloned_thread.id
    assert cloned_annotation.annotations == [{"id": "note-1"}]


@pytest.mark.asyncio
async def test_delete_project_preserves_shared_files_and_global_memory(
    lifecycle_sessionmaker,
    monkeypatch,
):
    now = utc_now()
    vector_db = SimpleNamespace(
        delete_thread_data=AsyncMock(return_value=True),
        delete_memory_vectors_for_scope=AsyncMock(return_value=True),
        delete_document_vectors_by_file_hash_and_model=AsyncMock(return_value=True),
    )
    monkeypatch.setattr(project_lifecycle_service, "get_vector_db", lambda: vector_db)
    checkpoint_cleanup = AsyncMock(return_value=["checkpoint-source"])
    monkeypatch.setattr(
        project_lifecycle_service,
        "delete_agent_checkpoints",
        checkpoint_cleanup,
    )
    delete_artifacts = AsyncMock(return_value=None)
    monkeypatch.setattr(project_lifecycle_service, "delete_file_artifacts", delete_artifacts)

    async with lifecycle_sessionmaker() as session:
        async with session.begin():
            session.add_all([
                Project(id="source-project", name="Source", embedding_model="BAAI/bge-m3"),
                Project(id="other-project", name="Other", embedding_model="BAAI/bge-m3"),
            ])
            session.add_all([
                File(file_hash="shared-file", file_name="shared.pdf", source_type="pdf"),
                File(file_hash="orphan-file", file_name="orphan.pdf", source_type="pdf"),
            ])
            await session.flush()
            session.add_all([
                ProjectFile(project_id="source-project", file_hash="shared-file"),
                ProjectFile(project_id="source-project", file_hash="orphan-file"),
                ProjectFile(project_id="other-project", file_hash="shared-file"),
            ])
            session.add(Thread(
                id="source-thread",
                project_id="source-project",
                name="Thread",
                embedding_model="BAAI/bge-m3",
            ))
            session.add(AgentWorkflow(
                id="workflow-1",
                name="Workflow 1",
                spec_json={},
                validation_result_json={},
            ))
            await session.flush()
            session.add(AgentRun(
                id="terminal-run",
                thread_id="source-thread",
                workflow_id="workflow-1",
                status="completed",
                checkpoint_thread_id="checkpoint-source",
                completed_at=now,
            ))
            session.add_all([
                _memory("project-memory", "project", "source-project", "Project"),
                _memory("thread-memory", "thread", "source-thread", "Thread"),
                _memory("global-memory", "user", "default", "Global"),
            ])

    result = await project_lifecycle_service.delete_project("source-project")

    assert result["deleted"] is True
    assert result["counts"]["canonical_files_deleted"] == 1
    async with lifecycle_sessionmaker() as session:
        assert await session.get(Project, "source-project") is None
        assert await session.get(Project, "other-project") is not None
        assert await session.get(File, "shared-file") is not None
        assert await session.get(File, "orphan-file") is None
        assert await session.get(Memory, "project-memory") is None
        assert await session.get(Memory, "thread-memory") is None
        assert await session.get(Memory, "global-memory") is not None

    vector_db.delete_thread_data.assert_awaited_once_with("source-thread")
    checkpoint_cleanup.assert_awaited_once_with(["checkpoint-source"])
    delete_artifacts.assert_awaited_once_with("orphan-file")


@pytest.mark.asyncio
async def test_lifecycle_summary_blocks_active_runs(lifecycle_sessionmaker):
    async with lifecycle_sessionmaker() as session:
        async with session.begin():
            session.add(Project(
                id="source-project",
                name="Source",
                embedding_model="BAAI/bge-m3",
            ))
            session.add(Thread(
                id="source-thread",
                project_id="source-project",
                name="Thread",
                embedding_model="BAAI/bge-m3",
            ))
            session.add(AgentWorkflow(
                id="workflow-1",
                name="Workflow 1",
                spec_json={},
                validation_result_json={},
            ))
            await session.flush()
            session.add(AgentRun(
                id="active-run",
                thread_id="source-thread",
                workflow_id="workflow-1",
                status="awaiting_human",
            ))

    summary = await project_lifecycle_service.get_project_lifecycle_summary("source-project")

    assert summary["active_run_count"] == 1
    assert summary["can_clone"] is False
    assert summary["can_delete"] is False
    with pytest.raises(project_lifecycle_service.ProjectActiveRunsError):
        await project_lifecycle_service.clone_project(
            "source-project",
            name="Copy",
            include_threads=False,
        )


@pytest.mark.asyncio
async def test_delete_cleanup_failure_preserves_project(
    lifecycle_sessionmaker,
    monkeypatch,
):
    vector_db = SimpleNamespace(
        delete_thread_data=AsyncMock(return_value=False),
        delete_memory_vectors_for_scope=AsyncMock(return_value=True),
        delete_document_vectors_by_file_hash_and_model=AsyncMock(return_value=True),
    )
    monkeypatch.setattr(project_lifecycle_service, "get_vector_db", lambda: vector_db)

    async with lifecycle_sessionmaker() as session:
        async with session.begin():
            session.add(Project(
                id="source-project",
                name="Source",
                embedding_model="BAAI/bge-m3",
            ))
            session.add(Thread(
                id="source-thread",
                project_id="source-project",
                name="Thread",
                embedding_model="BAAI/bge-m3",
            ))

    with pytest.raises(project_lifecycle_service.ProjectCleanupError):
        await project_lifecycle_service.delete_project("source-project")

    async with lifecycle_sessionmaker() as session:
        assert await session.get(Project, "source-project") is not None
        assert await session.get(Thread, "source-thread") is not None


@pytest.mark.asyncio
async def test_clone_memory_index_failure_keeps_retryable_clone(
    lifecycle_sessionmaker,
    monkeypatch,
):
    async def fail_index(memory):
        async with lifecycle_sessionmaker() as session:
            async with session.begin():
                stored = await session.get(Memory, memory.id)
                stored.index_status = "failed"
                stored.index_attempts = (stored.index_attempts or 0) + 1
                stored.index_error = "simulated indexing failure"
        raise RuntimeError("simulated indexing failure")

    monkeypatch.setattr(project_lifecycle_service, "index_memory_record", fail_index)
    async with lifecycle_sessionmaker() as session:
        async with session.begin():
            session.add(Project(
                id="source-project",
                name="Source",
                embedding_model="BAAI/bge-m3",
            ))
            session.add(_memory(
                "source-project-memory",
                "project",
                "source-project",
                "Project fact",
            ))

    result = await project_lifecycle_service.clone_project(
        "source-project",
        name="Copy",
        include_threads=False,
    )

    assert result["warnings"][0]["code"] == "memory_index_failed"
    memory_id = result["warnings"][0]["memory_id"]
    async with lifecycle_sessionmaker() as session:
        cloned_memory = await session.get(Memory, memory_id)
        assert cloned_memory is not None
        assert cloned_memory.scope_id == result["project"].id
        assert cloned_memory.index_status == "failed"
        assert cloned_memory.index_attempts == 1
