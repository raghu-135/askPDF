from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.enums import MemoryScopeType, MemoryType
from app.db.models_sqlmodel import ChatTurnStatus, Memory, MemoryEvent, Project
from app.db.repositories.memory_repo_sqlmodel import MemoryRepository
from app.db.repositories.message_repo_sqlmodel import MessageRepository
from app.db.repositories.project_file_repo_sqlmodel import ProjectFileRepository
from app.db.repositories.project_repo_sqlmodel import DEFAULT_PROJECT_NAME, ProjectRepository
from app.db.repositories.thread_repo_sqlmodel import ThreadRepository
from app.time_utils import utc_now
from app.services.memory_service import (
    _rank_fuse_memory_hits,
    memory_scope_policy_for_thread,
    search_thread_memory,
)
from app.services.memory_policy import (
    LOCAL_USER_MEMORY_SCOPE_ID,
    merge_project_settings_json,
    normalize_thread_memory_settings,
)
from app.db.vector.config import VectorDBInsertError


@pytest.fixture
def repo_sessionmaker(engine, monkeypatch):
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    import app.db.repositories.memory_repo_sqlmodel as memory_repo
    import app.db.repositories.message_repo_sqlmodel as message_repo
    import app.db.repositories.project_repo_sqlmodel as project_repo
    import app.db.repositories.project_file_repo_sqlmodel as project_file_repo
    import app.db.repositories.thread_repo_sqlmodel as thread_repo

    monkeypatch.setattr(memory_repo, "async_session_maker", maker)
    monkeypatch.setattr(message_repo, "async_session_maker", maker)
    monkeypatch.setattr(project_repo, "async_session_maker", maker)
    monkeypatch.setattr(project_file_repo, "async_session_maker", maker)
    monkeypatch.setattr(thread_repo, "async_session_maker", maker)
    return maker


@pytest.mark.asyncio
async def test_default_project_has_locked_embedding_model(repo_sessionmaker):
    project = await ProjectRepository().ensure_default_project()

    assert project.name == DEFAULT_PROJECT_NAME
    assert project.embedding_model == "BAAI/bge-m3"


@pytest.mark.asyncio
async def test_thread_repository_creates_thread_in_project(repo_sessionmaker):
    project = await ProjectRepository().create(
        name="Research",
        embedding_model="BAAI/bge-m3",
    )

    thread = await ThreadRepository().create("Thread", project.id)

    assert thread.project_id == project.id
    assert thread.embedding_model == project.embedding_model


@pytest.mark.asyncio
async def test_projects_sort_by_last_activity_and_activity_writes_are_monotonic(
    repo_sessionmaker,
    sample_file,
):
    project_repo = ProjectRepository()
    older = await project_repo.create(name="Older", embedding_model="BAAI/bge-m3")
    newer = await project_repo.create(name="Newer", embedding_model="BAAI/bge-m3")
    old_timestamp = utc_now() - timedelta(days=30)
    recent_timestamp = utc_now() - timedelta(days=1)

    async with repo_sessionmaker() as session:
        async with session.begin():
            older_row = await session.get(Project, older.id)
            newer_row = await session.get(Project, newer.id)
            older_row.last_activity_at = old_timestamp
            newer_row.last_activity_at = recent_timestamp

    assert [project.id for project in await project_repo.list_all()][:2] == [
        newer.id,
        older.id,
    ]

    thread = await ThreadRepository().create("Active thread", older.id)
    refreshed = await project_repo.get(older.id)
    assert refreshed.last_activity_at > recent_timestamp
    thread_activity = refreshed.last_activity_at

    await MessageRepository().create_turn(
        thread_id=thread.id,
        question="Do not reorder",
        status=ChatTurnStatus.CANCELLED.value,
        created_at=thread_activity + timedelta(days=1),
    )
    refreshed = await project_repo.get(older.id)
    assert refreshed.last_activity_at == thread_activity

    await MessageRepository().create_turn(
        thread_id=thread.id,
        question="Meaningful activity",
        answer="Completed",
        created_at=thread_activity + timedelta(days=2),
    )
    refreshed = await project_repo.get(older.id)
    assert refreshed.last_activity_at == thread_activity + timedelta(days=2)

    await ProjectFileRepository().add(older.id, sample_file.file_hash)
    file_activity = (await project_repo.get(older.id)).last_activity_at
    assert file_activity >= refreshed.last_activity_at

    memory = await MemoryRepository().create_memory(
        scope_type=MemoryScopeType.PROJECT.value,
        scope_id=older.id,
        memory_type=MemoryType.SEMANTIC.value,
        content="Project activity memory.",
        embedding_model="BAAI/bge-m3",
        content_hash="project-activity-memory",
    )
    memory_activity = (await project_repo.get(older.id)).last_activity_at
    assert memory_activity >= file_activity

    await MemoryRepository().delete_memory(memory.id)
    assert (await project_repo.get(older.id)).last_activity_at >= memory_activity


@pytest.mark.asyncio
async def test_project_memory_settings_preserve_unrelated_json(repo_sessionmaker):
    repo = ProjectRepository()
    project = await repo.create(
        name="Consent",
        embedding_model="BAAI/bge-m3",
        settings_json={
            "theme": "dense",
            "memory": {"project_reads_user_memory": False},
        },
    )

    updated = await repo.update(
        project.id,
        settings_json={"memory": {"project_reads_user_memory": True}},
    )

    assert updated.settings_json == {
        "theme": "dense",
        "memory": {"project_reads_user_memory": True},
    }


@pytest.mark.asyncio
async def test_memory_repository_index_lifecycle_and_audit(repo_sessionmaker):
    repo = MemoryRepository()

    memory = await repo.create_memory(
        scope_type=MemoryScopeType.PROJECT.value,
        scope_id="project-1",
        memory_type=MemoryType.SEMANTIC.value,
        content="The project codename is Atlas.",
        embedding_model="BAAI/bge-m3",
        content_hash="atlas-hash",
        confidence=0.9,
        created_by="test",
    )
    indexing = await repo.mark_memory_indexing(memory.id)
    indexed = await repo.mark_memory_indexed(memory.id)

    assert indexing.index_attempts == 1
    assert indexed.index_status == "indexed"
    assert indexed.indexed_at is not None
    async with repo_sessionmaker() as session:
        event_result = await session.execute(select(MemoryEvent).where(MemoryEvent.memory_id == memory.id))
        assert len(list(event_result.scalars().all())) == 1


@pytest.mark.asyncio
async def test_memory_repository_updates_same_canonical_record(repo_sessionmaker):
    repo = MemoryRepository()
    memory = await repo.create_memory(
        scope_type=MemoryScopeType.THREAD.value,
        scope_id="thread-update",
        memory_type=MemoryType.SEMANTIC.value,
        content="Old preference.",
        embedding_model="BAAI/bge-m3",
        content_hash="old-hash",
    )

    updated = await repo.update_memory(
        memory.id,
        memory_type=MemoryType.PROCEDURAL.value,
        content="New preference.",
        content_hash="new-hash",
        actor_id="curator",
        event_type="curator_updated",
    )

    assert updated.id == memory.id
    assert updated.memory_type == MemoryType.PROCEDURAL.value
    assert updated.content == "New preference."
    assert updated.index_status == "pending"
    async with repo_sessionmaker() as session:
        rows = list((await session.execute(
            select(Memory).where(Memory.id == memory.id)
        )).scalars().all())
        events = list((await session.execute(
            select(MemoryEvent).where(MemoryEvent.memory_id == memory.id)
        )).scalars().all())
    assert len(rows) == 1
    assert [event.event_type for event in events] == ["created", "curator_updated"]


@pytest.mark.asyncio
async def test_memory_index_failure_is_retryable_without_duplicate_canonical_record(
    repo_sessionmaker,
    monkeypatch,
):
    from app.services import memory_service

    repo = MemoryRepository()
    memory = await repo.create_memory(
        scope_type=MemoryScopeType.PROJECT.value,
        scope_id="project-1",
        memory_type=MemoryType.SEMANTIC.value,
        content="The project codename is Atlas.",
        embedding_model="BAAI/bge-m3",
        content_hash="atlas-retry-hash",
        confidence=0.9,
        created_by="test",
    )
    embedding_client = SimpleNamespace(
        aembed_query=AsyncMock(return_value=[0.1, 0.2, 0.3])
    )
    vector_db = SimpleNamespace(
        index_memory=AsyncMock(
            side_effect=[
                VectorDBInsertError("Weaviate rejected 1 of 1 model-aware batch objects"),
                1,
            ]
        )
    )
    monkeypatch.setattr(
        memory_service,
        "require_embedding_model_ready",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        memory_service,
        "get_embedding_model",
        lambda _model: embedding_client,
    )
    monkeypatch.setattr(memory_service, "get_vector_db", lambda: vector_db)

    with pytest.raises(VectorDBInsertError):
        await memory_service.index_memory_record(memory)

    failed = await repo.get_memory(memory.id)
    assert failed.index_status == "failed"
    assert failed.index_attempts == 1
    assert "Weaviate rejected 1 of 1" in failed.index_error
    assert failed.indexed_at is None

    await memory_service.index_memory_record(failed)

    indexed = await repo.get_memory(memory.id)
    assert indexed.index_status == "indexed"
    assert indexed.index_attempts == 2
    assert indexed.index_error is None
    assert indexed.indexed_at is not None
    async with repo_sessionmaker() as session:
        memories = (
            await session.execute(select(Memory).where(Memory.id == memory.id))
        ).scalars().all()
    assert [row.id for row in memories] == [memory.id]
    assert vector_db.index_memory.await_count == 2


@pytest.mark.asyncio
async def test_memory_repository_rejects_invalid_memory_values(repo_sessionmaker):
    repo = MemoryRepository()

    invalid_cases = [
        {"scope_type": "workspace", "scope_id": "scope-1", "memory_type": MemoryType.SEMANTIC.value, "content": "content"},
        {"scope_type": MemoryScopeType.THREAD.value, "scope_id": " ", "memory_type": MemoryType.SEMANTIC.value, "content": "content"},
        {"scope_type": MemoryScopeType.THREAD.value, "scope_id": "scope-1", "memory_type": "fact", "content": "content"},
        {"scope_type": MemoryScopeType.THREAD.value, "scope_id": "scope-1", "memory_type": MemoryType.SEMANTIC.value, "content": " "},
        {"scope_type": MemoryScopeType.THREAD.value, "scope_id": "scope-1", "memory_type": MemoryType.SEMANTIC.value, "content": "content", "confidence": 1.5},
        {"scope_type": MemoryScopeType.THREAD.value, "scope_id": "scope-1", "memory_type": MemoryType.SEMANTIC.value, "content": "content", "visibility": "public"},
    ]

    for kwargs in invalid_cases:
        kwargs.setdefault("embedding_model", "BAAI/bge-m3")
        kwargs.setdefault("content_hash", "content-hash")
        with pytest.raises(ValueError):
            await repo.create_memory(**kwargs)


@pytest.mark.asyncio
async def test_memory_repository_hard_deletes_memory_and_events(repo_sessionmaker):
    repo = MemoryRepository()

    memory = await repo.create_memory(
        scope_type=MemoryScopeType.THREAD.value,
        scope_id="thread-1",
        memory_type=MemoryType.SEMANTIC.value,
        content="Delete this durable memory.",
        embedding_model="BAAI/bge-m3",
        content_hash="delete-hash",
        created_by="tester",
    )
    deleted = await repo.delete_memory(memory.id)

    assert deleted is True
    assert await repo.get_memory(memory.id) is None
    async with repo_sessionmaker() as session:
        event_result = await session.execute(select(MemoryEvent).where(MemoryEvent.memory_id == memory.id))
        assert list(event_result.scalars().all()) == []


@pytest.mark.asyncio
async def test_memory_repository_deletes_scope_and_expired_memories(repo_sessionmaker):
    repo = MemoryRepository()
    now = utc_now()

    scoped = await repo.create_memory(
        scope_type=MemoryScopeType.THREAD.value,
        scope_id="thread-cleanup",
        memory_type=MemoryType.SEMANTIC.value,
        content="Thread cleanup memory.",
        embedding_model="BAAI/bge-m3",
        content_hash="thread-cleanup-hash",
    )
    expired = await repo.create_memory(
        scope_type=MemoryScopeType.PROJECT.value,
        scope_id="project-cleanup",
        memory_type=MemoryType.SEMANTIC.value,
        content="Expired cleanup memory.",
        embedding_model="BAAI/bge-m3",
        content_hash="expired-hash",
        expires_at=now - timedelta(minutes=1),
    )
    fresh = await repo.create_memory(
        scope_type=MemoryScopeType.PROJECT.value,
        scope_id="project-cleanup",
        memory_type=MemoryType.SEMANTIC.value,
        content="Fresh memory.",
        embedding_model="BAAI/bge-m3",
        content_hash="fresh-hash",
        expires_at=now + timedelta(minutes=1),
    )

    scoped_ids = await repo.delete_memories_for_scope(
        scope_type=MemoryScopeType.THREAD.value,
        scope_id="thread-cleanup",
    )
    expired_rows = await repo.delete_expired_memories(now=now)

    assert scoped_ids == [scoped.id]
    assert [memory.id for memory in expired_rows] == [expired.id]
    async with repo_sessionmaker() as session:
        remaining = (await session.execute(select(Memory))).scalars().all()
        assert [memory.id for memory in remaining] == [fresh.id]


def test_memory_setting_normalization_removes_legacy_flags():
    assert normalize_thread_memory_settings({
        "memory": {
            "global_memory_enabled": True,
            "thread_reads_project_memory": False,
            "thread_reads_user_memory": True,
            "project_reads_user_memory": True,
        }
    }) == {
        "thread_reads_project_memory": False,
        "thread_reads_user_memory": True,
    }
    assert normalize_thread_memory_settings({
        "memory": {
            "global_memory_enabled": False,
            "thread_reads_user_memory": True,
        }
    })["thread_reads_user_memory"] is False
    assert merge_project_settings_json(
        {"other": "kept", "memory": {"global_memory_enabled": True}},
    ) == {
        "other": "kept",
        "memory": {"project_reads_user_memory": False},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("project_gate", "thread_gate", "expects_user", "skip_reason"),
    [
        (True, True, True, None),
        (False, True, False, "project_opt_out"),
        (True, False, False, "thread_opt_out"),
        (False, False, False, "project_opt_out"),
    ],
)
async def test_global_memory_requires_project_and_thread_consent(
    monkeypatch,
    project_gate,
    thread_gate,
    expects_user,
    skip_reason,
):
    from app.services import memory_service

    context = SimpleNamespace(
        thread=SimpleNamespace(
            id="thread-1",
            settings={
                "memory": {
                    "thread_reads_project_memory": True,
                    "thread_reads_user_memory": thread_gate,
                }
            },
        ),
        project=SimpleNamespace(
            id="project-1",
            settings_json={
                "memory": {"project_reads_user_memory": project_gate}
            },
        ),
    )
    monkeypatch.setattr(
        memory_service,
        "resolve_thread_embedding_context",
        AsyncMock(return_value=context),
    )

    policy = await memory_scope_policy_for_thread("thread-1")

    assert {
        (scope["scope_type"], scope["scope_id"])
        for scope in policy["searched_scopes"]
    } == {
        ("thread", "thread-1"),
        ("project", "project-1"),
        *(([("user", LOCAL_USER_MEMORY_SCOPE_ID)] if expects_user else [])),
    }
    user_skips = [
        item for item in policy["skipped_scopes"] if item["scope_type"] == "user"
    ]
    if expects_user:
        assert user_skips == []
    else:
        assert user_skips == [{"scope_type": "user", "reason": skip_reason}]


@pytest.mark.asyncio
async def test_explicit_scope_filter_is_upper_bound_and_empty_means_none(monkeypatch):
    from app.services import memory_service

    context = SimpleNamespace(
        thread=SimpleNamespace(
            id="thread-1",
            settings={
                "memory": {
                    "thread_reads_project_memory": True,
                    "thread_reads_user_memory": False,
                }
            },
        ),
        project=SimpleNamespace(
            id="project-1",
            settings_json={
                "memory": {"project_reads_user_memory": True}
            },
        ),
    )
    monkeypatch.setattr(
        memory_service,
        "resolve_thread_embedding_context",
        AsyncMock(return_value=context),
    )

    unrestricted = await memory_scope_policy_for_thread("thread-1", None)
    empty = await memory_scope_policy_for_thread("thread-1", [])
    user_only = await memory_scope_policy_for_thread("thread-1", ["user"])

    assert unrestricted["requested_scopes"] == ["thread", "project", "user"]
    assert [scope["scope_type"] for scope in unrestricted["searched_scopes"]] == [
        "thread",
        "project",
    ]
    assert empty["requested_scopes"] == []
    assert empty["searched_scopes"] == []
    assert user_only["searched_scopes"] == []
    assert user_only["skipped_scopes"][-1] == {
        "scope_type": "user",
        "reason": "thread_opt_out",
    }


@pytest.mark.asyncio
async def test_normal_recall_fuses_project_and_default_user_memory(monkeypatch):
    from app.services import memory_service

    context = SimpleNamespace(
        embedding_model="project-embedding-model",
        thread=SimpleNamespace(
            id="thread-1",
            settings={
                "memory": {
                    "thread_reads_project_memory": True,
                    "thread_reads_user_memory": True,
                }
            },
        ),
        project=SimpleNamespace(
            id="project-1",
            settings_json={
                "memory": {"project_reads_user_memory": True}
            },
        ),
    )
    vector_db = SimpleNamespace(search_memory=AsyncMock())

    async def fake_search_memory(*, embedding_model, **_kwargs):
        if embedding_model == "project-embedding-model":
            return [{
                "memory_id": "project-memory",
                "scope_type": "project",
                "score": 0.77,
            }]
        return [{
            "memory_id": "user-memory",
            "scope_type": "user",
            "score": 0.96,
        }]

    vector_db.search_memory.side_effect = fake_search_memory
    memories = {
        "project-memory": SimpleNamespace(
            id="project-memory",
            scope_type="project",
            scope_id="project-1",
            memory_type="semantic",
            content="Project fact",
            summary="",
            source_refs_json={},
            confidence=0.9,
            visibility="project",
            status="active",
            expires_at=None,
            created_at=None,
            updated_at=None,
        ),
        "user-memory": SimpleNamespace(
            id="user-memory",
            scope_type="user",
            scope_id=LOCAL_USER_MEMORY_SCOPE_ID,
            memory_type="semantic",
            content="Use concise answers",
            summary="",
            source_refs_json={},
            confidence=0.9,
            visibility="private",
            status="active",
            expires_at=None,
            created_at=None,
            updated_at=None,
        ),
    }
    monkeypatch.setattr(
        memory_service,
        "resolve_thread_embedding_context",
        AsyncMock(return_value=context),
    )
    monkeypatch.setattr(
        memory_service,
        "require_embedding_model_ready",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        memory_service,
        "get_embedding_model",
        lambda model: SimpleNamespace(
            aembed_query=AsyncMock(return_value=[float(len(model))])
        ),
    )
    monkeypatch.setattr(memory_service, "get_vector_db", lambda: vector_db)
    monkeypatch.setattr(
        memory_service,
        "get_memory",
        AsyncMock(side_effect=lambda memory_id: memories[memory_id]),
    )

    result = await search_thread_memory(
        thread_id="thread-1",
        query="How should you answer?",
    )

    assert [item["id"] for item in result["memories"]] == [
        "project-memory",
        "user-memory",
    ]
    assert result["memories"][1]["scope_id"] == LOCAL_USER_MEMORY_SCOPE_ID
    assert result["memories"][1]["embedding_model"] == memory_service.GLOBAL_MEMORY_EMBEDDING_MODEL
    assert all(item["score_type"] == "rrf" for item in result["memories"])
    assert vector_db.search_memory.await_count == 2


def test_memory_rank_fusion_does_not_compare_raw_scores_across_models():
    fused = _rank_fuse_memory_hits(
        [
            (
                "project-model",
                [
                    {
                        "memory_id": "project-rank-1",
                        "scope_type": "project",
                        "score": 0.05,
                    },
                    {
                        "memory_id": "project-rank-2",
                        "scope_type": "project",
                        "score": 0.99,
                    },
                ],
            ),
            (
                "BAAI/bge-m3",
                [
                    {
                        "memory_id": "user-rank-1",
                        "scope_type": "user",
                        "score": 0.999,
                    },
                ],
            ),
        ]
    )

    assert [hit["memory_id"] for hit in fused] == [
        "project-rank-1",
        "user-rank-1",
        "project-rank-2",
    ]
    assert fused[0]["score"] == fused[1]["score"]
    assert fused[0]["raw_score"] == 0.05
    assert fused[1]["raw_score"] == 0.999
    assert fused[0]["embedding_model"] == "project-model"
    assert fused[1]["embedding_model"] == "BAAI/bge-m3"
