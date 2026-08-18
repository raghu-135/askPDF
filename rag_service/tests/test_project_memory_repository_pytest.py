from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.enums import MemoryScopeType
from app.db.models_sqlmodel import (
    ChatTurnStatus,
    EmbeddingJob,
    GlobalMemoryRepresentation,
    Memory,
    MemoryEvent,
    MemoryOverride,
    Project,
)
from app.db.repositories.memory_repo_sqlmodel import MemoryRepository
from app.db.repositories.message_repo_sqlmodel import MessageRepository
from app.db.repositories.project_file_repo_sqlmodel import ProjectFileRepository
from app.db.repositories.project_repo_sqlmodel import DEFAULT_PROJECT_NAME, ProjectRepository
from app.db.repositories.thread_repo_sqlmodel import ThreadRepository
from app.time_utils import utc_now
from app.services.memory_service import (
    _merge_same_model_memory_hits,
    memory_scope_policy_for_thread,
    search_thread_memory,
)
from app.services.effective_memory_service import resolve_effective_memory_context
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
    import app.services.effective_memory_service as effective_memory_service
    import app.services.embedding_materialization_service as embedding_materialization_service

    monkeypatch.setattr(memory_repo, "async_session_maker", maker)
    monkeypatch.setattr(message_repo, "async_session_maker", maker)
    monkeypatch.setattr(project_repo, "async_session_maker", maker)
    monkeypatch.setattr(project_file_repo, "async_session_maker", maker)
    monkeypatch.setattr(thread_repo, "async_session_maker", maker)
    monkeypatch.setattr(effective_memory_service, "async_session_maker", maker)
    monkeypatch.setattr(embedding_materialization_service, "async_session_maker", maker)
    return maker


@pytest.mark.asyncio
async def test_embedding_jobs_are_deduplicated_and_refresh_source_version(repo_sessionmaker):
    from app.services.embedding_materialization_service import ensure_embedding_job

    first = await ensure_embedding_job(
        resource_type="global_memory",
        resource_id="memory-job-1",
        scope_id="default",
        embedding_model="text-embedding-nomic-embed-text-v1.5",
        source_version="hash-a",
    )
    second = await ensure_embedding_job(
        resource_type="global_memory",
        resource_id="memory-job-1",
        scope_id="default",
        embedding_model="text-embedding-nomic-embed-text-v1.5",
        source_version="hash-b",
    )

    assert first.id == second.id
    async with repo_sessionmaker() as session:
        rows = list((await session.execute(select(EmbeddingJob))).scalars().all())
    assert len(rows) == 1
    assert rows[0].source_version == "hash-b"
    assert rows[0].status == "pending"


@pytest.mark.asyncio
async def test_global_model_backfill_creates_only_missing_representations(repo_sessionmaker):
    from app.services.embedding_materialization_service import enqueue_global_model_jobs

    async with repo_sessionmaker() as session:
        async with session.begin():
            session.add(Project(id="job-project-a", name="A", embedding_model="model-a"))
            session.add(Project(id="job-project-b", name="B", embedding_model="model-b"))
            memory = Memory(
                id="job-memory-1",
                scope_type=MemoryScopeType.USER.value,
                scope_id=LOCAL_USER_MEMORY_SCOPE_ID,
                content="A durable preference",
                embedding_model="BAAI/bge-m3",
                content_hash="job-hash",
            )
            session.add(memory)

    assert await enqueue_global_model_jobs("model-a") == 1
    assert await enqueue_global_model_jobs("model-a") == 1

    async with repo_sessionmaker() as session:
        reps = list((await session.execute(select(GlobalMemoryRepresentation))).scalars().all())
        jobs = list((await session.execute(select(EmbeddingJob))).scalars().all())
    assert {(row.memory_id, row.embedding_model) for row in reps} == {("job-memory-1", "model-a")}
    assert {(row.resource_id, row.embedding_model) for row in jobs} == {("job-memory-1", "model-a")}


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
        content="The project codename is Atlas.",
        embedding_model="BAAI/bge-m3",
        content_hash="atlas-hash",
        actor_id="test",
    )
    semantic_before_indexing = memory.semantic_updated_at
    indexing = await repo.mark_memory_indexing(memory.id)
    indexed = await repo.mark_memory_indexed(memory.id)

    assert indexing.index_attempts == 1
    assert indexed.index_status == "indexed"
    assert indexed.indexed_at is not None
    assert indexed.semantic_updated_at == semantic_before_indexing
    async with repo_sessionmaker() as session:
        event_result = await session.execute(select(MemoryEvent).where(MemoryEvent.memory_id == memory.id))
        assert len(list(event_result.scalars().all())) == 1


@pytest.mark.asyncio
async def test_memory_repository_updates_same_canonical_record(repo_sessionmaker):
    repo = MemoryRepository()
    memory = await repo.create_memory(
        scope_type=MemoryScopeType.THREAD.value,
        scope_id="thread-update",
        content="Old preference.",
        embedding_model="BAAI/bge-m3",
        content_hash="old-hash",
    )

    updated = await repo.update_memory(
        memory.id,
        content="New preference.",
        content_hash="new-hash",
        actor_id="curator",
        event_type="curator_updated",
    )

    assert updated.id == memory.id
    assert updated.content == "New preference."
    assert updated.index_status == "pending"
    assert updated.semantic_updated_at > memory.semantic_updated_at
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
        content="The project codename is Atlas.",
        embedding_model="BAAI/bge-m3",
        content_hash="atlas-retry-hash",
        actor_id="test",
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
        {"scope_type": "workspace", "scope_id": "scope-1", "content": "content"},
        {"scope_type": MemoryScopeType.THREAD.value, "scope_id": " ", "content": "content"},
        {"scope_type": MemoryScopeType.THREAD.value, "scope_id": "scope-1", "content": " "},
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
        content="Delete this durable memory.",
        embedding_model="BAAI/bge-m3",
        content_hash="delete-hash",
        actor_id="tester",
    )
    deleted = await repo.delete_memory(memory.id)

    assert deleted is True
    assert await repo.get_memory(memory.id) is None
    async with repo_sessionmaker() as session:
        event_result = await session.execute(select(MemoryEvent).where(MemoryEvent.memory_id == memory.id))
        assert list(event_result.scalars().all()) == []


@pytest.mark.asyncio
async def test_memory_repository_deletes_only_requested_scope(repo_sessionmaker):
    repo = MemoryRepository()

    scoped = await repo.create_memory(
        scope_type=MemoryScopeType.THREAD.value,
        scope_id="thread-cleanup",
        content="Thread cleanup memory.",
        embedding_model="BAAI/bge-m3",
        content_hash="thread-cleanup-hash",
    )
    retained = await repo.create_memory(
        scope_type=MemoryScopeType.PROJECT.value,
        scope_id="project-cleanup",
        content="Retained memory.",
        embedding_model="BAAI/bge-m3",
        content_hash="retained-hash",
    )

    scoped_ids = await repo.delete_memories_for_scope(
        scope_type=MemoryScopeType.THREAD.value,
        scope_id="thread-cleanup",
    )
    assert scoped_ids == [scoped.id]
    async with repo_sessionmaker() as session:
        remaining = (await session.execute(select(Memory))).scalars().all()
        assert [memory.id for memory in remaining] == [retained.id]


def test_memory_setting_normalization_removes_legacy_flags():
    assert normalize_thread_memory_settings({
        "memory": {
            "global_memory_enabled": True,
            "thread_reads_project_memory": False,
            "thread_reads_user_memory": True,
            "project_reads_user_memory": True,
        }
    }) == {
        "memory_enabled": True,
        "thread_reads_thread_memory": True,
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
    repo_sessionmaker,
    project_gate,
    thread_gate,
    expects_user,
    skip_reason,
):
    project = await ProjectRepository().create(
        name="Consent matrix",
        embedding_model="BAAI/bge-m3",
        settings_json={"memory": {"project_reads_user_memory": project_gate}},
    )
    thread = await ThreadRepository().create("Consent thread", project.id)
    await ThreadRepository().update_settings(
        thread.id,
        {"memory": {
            "thread_reads_project_memory": True,
            "thread_reads_user_memory": thread_gate,
        }},
    )

    policy = await memory_scope_policy_for_thread(thread.id)

    assert {
        (scope["scope_type"], scope["scope_id"])
        for scope in policy["searched_scopes"]
    } == {
        ("thread", thread.id),
        ("project", project.id),
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
async def test_explicit_scope_filter_is_upper_bound_and_empty_means_none(repo_sessionmaker):
    project = await ProjectRepository().create(
        name="Explicit scopes",
        embedding_model="BAAI/bge-m3",
        settings_json={"memory": {"project_reads_user_memory": True}},
    )
    thread = await ThreadRepository().create("Explicit scope thread", project.id)
    await ThreadRepository().update_settings(
        thread.id,
        {"memory": {
            "thread_reads_project_memory": True,
            "thread_reads_user_memory": False,
        }},
    )

    unrestricted = await memory_scope_policy_for_thread(thread.id, None)
    empty = await memory_scope_policy_for_thread(thread.id, [])
    user_only = await memory_scope_policy_for_thread(thread.id, ["user"])

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
async def test_master_and_thread_local_recall_switches_are_independent(repo_sessionmaker):
    project = await ProjectRepository().create(
        name="Recall switches",
        embedding_model="BAAI/bge-m3",
    )
    thread = await ThreadRepository().create("Recall switch thread", project.id)
    await ThreadRepository().update_settings(
        thread.id,
        {"memory": {
            "memory_enabled": False,
            "thread_reads_thread_memory": True,
            "thread_reads_project_memory": True,
        }},
    )

    disabled = await memory_scope_policy_for_thread(thread.id)
    assert disabled["searched_scopes"] == []
    assert {item["reason"] for item in disabled["skipped_scopes"]} == {"memory_disabled"}

    await ThreadRepository().update_settings(
        thread.id,
        {"memory": {
            "memory_enabled": True,
            "thread_reads_thread_memory": False,
            "thread_reads_project_memory": True,
        }},
    )
    thread_opt_out = await memory_scope_policy_for_thread(thread.id)
    assert [item["scope_type"] for item in thread_opt_out["searched_scopes"]] == ["project"]
    assert {item["scope_type"]: item["reason"] for item in thread_opt_out["skipped_scopes"]}["thread"] == "thread_opt_out"


@pytest.mark.asyncio
async def test_normal_recall_merges_project_and_default_user_memory_in_one_model(monkeypatch):
    from app.services import memory_service

    context = SimpleNamespace(
        embedding_model="BAAI/bge-m3",
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

    async def fake_search_memory(*, embedding_model, **kwargs):
        assert [scope["scope_type"] for scope in kwargs["scope_filters"]] == ["thread", "project", "user"]
        return [{
                "memory_id": "project-memory",
                "scope_type": "project",
                "score": 0.77,
            }, {
                "memory_id": "user-memory",
                "scope_type": "user",
                "score": 0.96,
            }, {
                "memory_id": "weak-thread-memory",
                "scope_type": "thread",
                "score": 0.30,
            }]

    vector_db.search_memory.side_effect = fake_search_memory
    memories = {
        "project-memory": SimpleNamespace(
            id="project-memory",
            scope_type="project",
            scope_id="project-1",
            content="Project fact",
            source_refs_json={},
            embedding_model="BAAI/bge-m3",
            content_hash="project-hash",
            index_status="indexed",
            index_attempts=1,
            indexed_at=None,
            index_error=None,
            created_at=None,
            updated_at=None,
        ),
        "user-memory": SimpleNamespace(
            id="user-memory",
            scope_type="user",
            scope_id=LOCAL_USER_MEMORY_SCOPE_ID,
            content="Use concise answers",
            source_refs_json={},
            embedding_model=memory_service.GLOBAL_MEMORY_EMBEDDING_MODEL,
            content_hash="user-hash",
            index_status="indexed",
            index_attempts=1,
            indexed_at=None,
            index_error=None,
            created_at=None,
            updated_at=None,
        ),
        "weak-thread-memory": SimpleNamespace(
            id="weak-thread-memory",
            scope_type="thread",
            scope_id="thread-1",
            content="A weakly related fact",
            source_refs_json={},
            embedding_model="BAAI/bge-m3",
            content_hash="weak-thread-hash",
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
        "resolve_effective_memory_context",
        AsyncMock(return_value={
            "policy": {
                "requested_scopes": ["thread", "project", "user"],
                "searched_scopes": [
                    {"scope_type": "thread", "scope_id": "thread-1"},
                    {"scope_type": "project", "scope_id": "project-1"},
                    {"scope_type": "user", "scope_id": LOCAL_USER_MEMORY_SCOPE_ID},
                ],
                "skipped_scopes": [],
            },
            "memory_records": list(memories.values()),
            "applied_overrides": [],
            "suppressed_memory_ids": [],
            "excluded_memory_ids": ["unavailable-memory"],
            "unavailable_memory_count": 0,
        }),
    )

    result = await search_thread_memory(
        thread_id="thread-1",
        query="How should you answer?",
    )

    assert [item["id"] for item in result["memories"]] == [
        "user-memory",
        "project-memory",
    ]
    assert result["memories"][0]["scope_id"] == LOCAL_USER_MEMORY_SCOPE_ID
    assert result["memories"][0]["embedding_model"] == memory_service.GLOBAL_MEMORY_EMBEDDING_MODEL
    assert all(item["score_type"] == "similarity" for item in result["memories"])
    assert result["retrieval_debug"]["rejection_reasons"]["below_relevance_threshold"] == 1
    assert result["retrieval_debug"]["recalled_ids"] == ["user-memory", "project-memory"]
    # All eligible scopes share one query embedding and one multi-scope vector query.
    assert vector_db.search_memory.await_count == 1
    assert all(
        call.kwargs["excluded_memory_ids"] == ["unavailable-memory"]
        for call in vector_db.search_memory.await_args_list
    )


@pytest.mark.asyncio
async def test_normal_recall_never_falls_back_when_global_representation_is_missing(monkeypatch):
    from app.services import embedding_materialization_service, memory_service

    class EmptyScalars:
        def all(self):
            return []

    class EmptyResult:
        def scalars(self):
            return EmptyScalars()

    class EmptySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def execute(self, _query):
            return EmptyResult()

    project_model = "project-embedding-model"
    context = SimpleNamespace(embedding_model=project_model)
    global_memory = SimpleNamespace(
        id="global-memory",
        scope_type="user",
        scope_id=LOCAL_USER_MEMORY_SCOPE_ID,
        content="Global preference",
        source_refs_json={},
        embedding_model=memory_service.GLOBAL_MEMORY_EMBEDDING_MODEL,
        content_hash="global-hash",
        created_at=None,
        updated_at=None,
    )
    vector_db = SimpleNamespace(search_memory=AsyncMock(return_value=[]))
    enqueue = AsyncMock(return_value=1)
    drain = AsyncMock(return_value=0)
    monkeypatch.setattr(memory_service, "async_session_maker", lambda: EmptySession())
    monkeypatch.setattr(memory_service, "resolve_thread_embedding_context", AsyncMock(return_value=context))
    monkeypatch.setattr(memory_service, "resolve_effective_memory_context", AsyncMock(return_value={
        "policy": {
            "requested_scopes": ["user"],
            "searched_scopes": [{"scope_type": "user", "scope_id": LOCAL_USER_MEMORY_SCOPE_ID}],
            "skipped_scopes": [],
        },
        "memory_records": [global_memory],
        "applied_overrides": [],
        "suppressed_memory_ids": [],
        "excluded_memory_ids": [],
        "unavailable_memory_count": 0,
    }))
    monkeypatch.setattr(memory_service, "require_embedding_model_ready", AsyncMock(return_value=None))
    monkeypatch.setattr(
        memory_service,
        "get_embedding_model",
        lambda _model: SimpleNamespace(aembed_query=AsyncMock(return_value=[0.1])),
    )
    monkeypatch.setattr(memory_service, "get_vector_db", lambda: vector_db)
    monkeypatch.setattr(embedding_materialization_service, "enqueue_global_model_jobs", enqueue)
    monkeypatch.setattr(embedding_materialization_service, "drain_embedding_jobs", drain)

    result = await memory_service.search_thread_memory(thread_id="thread-1", query="preference")
    await asyncio.sleep(0)

    assert result["memories"] == []
    assert result["representation_issues"] == [{
        "scope_type": "user",
        "embedding_model": project_model,
        "missing_count": 1,
        "reason": "global_representation_warming",
    }]
    assert {call.kwargs["embedding_model"] for call in vector_db.search_memory.await_args_list} == {project_model}
    enqueue.assert_awaited_once_with(project_model)


@pytest.mark.asyncio
async def test_override_set_replacement_is_audited(repo_sessionmaker):
    repo = MemoryRepository()
    source = await repo.create_memory(
        scope_type="thread",
        scope_id="thread-relations",
        content="Thread preference.",
        embedding_model="BAAI/bge-m3",
        content_hash="thread-relations-source",
    )
    first = await repo.create_memory(
        scope_type="project",
        scope_id="project-relations",
        content="Project preference.",
        embedding_model="BAAI/bge-m3",
        content_hash="thread-relations-first",
    )
    second = await repo.create_memory(
        scope_type="user",
        scope_id=LOCAL_USER_MEMORY_SCOPE_ID,
        content="Global preference.",
        embedding_model="BAAI/bge-m3",
        content_hash="thread-relations-second",
    )

    async with repo_sessionmaker() as session:
        async with session.begin():
            transactional = MemoryRepository(session)
            await transactional.replace_overrides(
                source.id,
                [first.id, second.id, first.id],
                actor_id="curator",
            )
            await transactional.replace_overrides(source.id, [second.id], actor_id="curator")

    edges = await repo.list_override_edges(memory_ids=[source.id])
    assert [(edge.overriding_memory_id, edge.overridden_memory_id) for edge in edges] == [
        (source.id, second.id),
    ]
    async with repo_sessionmaker() as session:
        events = list((await session.execute(
            select(MemoryEvent).where(
                MemoryEvent.memory_id == source.id,
                MemoryEvent.event_type == "override_set",
            )
        )).scalars().all())
    assert [event.payload_json["overridden_memory_ids"] for event in events] == [
        sorted([first.id, second.id]),
        [second.id],
    ]


@pytest.mark.asyncio
async def test_effective_view_applies_indexed_overrides_and_restores_on_delete(repo_sessionmaker):
    project = await ProjectRepository().create(
        name="Effective memory",
        embedding_model="BAAI/bge-m3",
        settings_json={"memory": {"project_reads_user_memory": True}},
    )
    thread = await ThreadRepository().create("Effective thread", project.id)
    await ThreadRepository().update_settings(
        thread.id,
        {"memory": {
            "thread_reads_project_memory": True,
            "thread_reads_user_memory": True,
        }},
    )
    repo = MemoryRepository()
    global_memory = await repo.create_memory(
        scope_type="user",
        scope_id=LOCAL_USER_MEMORY_SCOPE_ID,
        content="Use concise answers.",
        embedding_model="BAAI/bge-m3",
        content_hash="effective-global",
    )
    project_memory = await repo.create_memory(
        scope_type="project",
        scope_id=project.id,
        content="Use detailed answers in this project.",
        embedding_model="BAAI/bge-m3",
        content_hash="effective-project",
    )
    thread_memory = await repo.create_memory(
        scope_type="thread",
        scope_id=thread.id,
        content="Use bullet points in this thread.",
        embedding_model="BAAI/bge-m3",
        content_hash="effective-thread",
    )
    await repo.mark_memory_indexed(global_memory.id)
    await repo.mark_memory_indexed(project_memory.id)
    await repo.mark_memory_index_failed(thread_memory.id, "offline")
    async with repo_sessionmaker() as session:
        async with session.begin():
            transactional = MemoryRepository(session)
            await transactional.replace_overrides(project_memory.id, [global_memory.id])
            await transactional.replace_overrides(thread_memory.id, [project_memory.id])

    failed_view = await resolve_effective_memory_context(thread_id=thread.id)
    assert [memory["id"] for memory in failed_view["memories"]] == [project_memory.id]
    assert failed_view["suppressed_memory_ids"] == [global_memory.id]
    assert failed_view["unavailable_memory_count"] == 1
    assert [section["scope_type"] for section in failed_view["workspace_sections"]] == [
        "thread", "project", "user",
    ]
    failed_rows = {
        memory["id"]: memory
        for section in failed_view["workspace_sections"]
        for memory in section["memories"]
    }
    assert failed_rows[thread_memory.id]["resolution_status"] == "unavailable"
    assert failed_rows[project_memory.id]["resolution_status"] == "effective"
    assert failed_rows[global_memory.id]["resolution_status"] == "overridden"
    assert failed_rows[thread_memory.id]["applied_overrides"] == []
    assert [item["id"] for item in failed_rows[project_memory.id]["applied_overrides"]] == [global_memory.id]

    await repo.mark_memory_indexed(thread_memory.id)
    indexed_view = await resolve_effective_memory_context(thread_id=thread.id)
    assert [memory["id"] for memory in indexed_view["memories"]] == [thread_memory.id]
    assert set(indexed_view["suppressed_memory_ids"]) == {
        global_memory.id,
        project_memory.id,
    }
    indexed_rows = {
        memory["id"]: memory
        for section in indexed_view["workspace_sections"]
        for memory in section["memories"]
    }
    assert indexed_rows[thread_memory.id]["resolution_status"] == "effective"
    assert indexed_rows[project_memory.id]["resolution_status"] == "overridden"
    assert indexed_rows[global_memory.id]["resolution_status"] == "overridden"
    assert [item["id"] for item in indexed_rows[project_memory.id]["applied_overridden_by"]] == [thread_memory.id]

    sibling = await ThreadRepository().create("Sibling thread", project.id)
    await ThreadRepository().update_settings(
        sibling.id,
        {"memory": {
            "thread_reads_project_memory": True,
            "thread_reads_user_memory": True,
        }},
    )
    sibling_view = await resolve_effective_memory_context(thread_id=sibling.id)
    sibling_rows = {
        memory["id"]: memory
        for section in sibling_view["workspace_sections"]
        for memory in section["memories"]
    }
    assert sibling_rows[project_memory.id]["resolution_status"] == "effective"
    assert sibling_rows[project_memory.id]["overridden_by"] == []
    assert sibling_rows[project_memory.id]["applied_overridden_by"] == []
    assert [item["id"] for item in sibling_rows[global_memory.id]["overridden_by"]] == [project_memory.id]

    project_view = await resolve_effective_memory_context(project_id=project.id)
    assert [memory["id"] for memory in project_view["memories"]] == [project_memory.id]
    assert project_view["suppressed_memory_ids"] == [global_memory.id]
    assert [section["scope_type"] for section in project_view["workspace_sections"]] == ["project", "user"]

    await ThreadRepository().update_settings(
        thread.id,
        {"memory": {
            "thread_reads_project_memory": False,
            "thread_reads_user_memory": False,
        }},
    )
    disabled_view = await resolve_effective_memory_context(thread_id=thread.id)
    disabled_sections = {section["scope_type"]: section for section in disabled_view["workspace_sections"]}
    assert disabled_sections["project"]["recall_enabled"] is False
    assert disabled_sections["project"]["recall_skip_reason"] == "thread_opt_out"
    assert disabled_sections["project"]["memories"][0]["resolution_status"] == "recall_disabled"
    assert disabled_sections["user"]["memories"][0]["resolution_status"] == "recall_disabled"
    assert [memory["id"] for memory in disabled_view["memories"]] == [thread_memory.id]

    await ThreadRepository().update_settings(
        thread.id,
        {"memory": {
            "thread_reads_project_memory": True,
            "thread_reads_user_memory": True,
        }},
    )
    await repo.delete_memory(thread_memory.id)
    restored_view = await resolve_effective_memory_context(thread_id=thread.id)
    assert [memory["id"] for memory in restored_view["memories"]] == [project_memory.id]


@pytest.mark.asyncio
async def test_workspace_sections_apply_limits_independently(repo_sessionmaker):
    project = await ProjectRepository().create(
        name="Limited memory",
        embedding_model="BAAI/bge-m3",
        settings_json={"memory": {"project_reads_user_memory": True}},
    )
    repo = MemoryRepository()
    for index in range(2):
        project_memory = await repo.create_memory(
            scope_type="project",
            scope_id=project.id,
            content=f"Project memory {index}",
            embedding_model="BAAI/bge-m3",
            content_hash=f"limited-project-{index}",
        )
        global_memory = await repo.create_memory(
            scope_type="user",
            scope_id=LOCAL_USER_MEMORY_SCOPE_ID,
            content=f"Global memory {index}",
            embedding_model="BAAI/bge-m3",
            content_hash=f"limited-global-{index}",
        )
        await repo.mark_memory_indexed(project_memory.id)
        await repo.mark_memory_indexed(global_memory.id)

    view = await resolve_effective_memory_context(project_id=project.id, limit=1)
    assert len(view["memories"]) == 1
    assert [section["scope_type"] for section in view["workspace_sections"]] == ["project", "user"]
    assert [len(section["memories"]) for section in view["workspace_sections"]] == [1, 1]
    assert [section["truncated"] for section in view["workspace_sections"]] == [True, True]


def test_same_model_memory_merge_uses_comparable_similarity_scores():
    merged = _merge_same_model_memory_hits(
        "project-model",
        [[
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
                ], [
                    {
                        "memory_id": "user-rank-1",
                        "scope_type": "user",
                        "score": 0.999,
                    },
                ]]
    )

    assert [hit["memory_id"] for hit in merged] == [
        "user-rank-1",
        "project-rank-2",
        "project-rank-1",
    ]
    assert merged[0]["raw_score"] == 0.999
    assert all(hit["embedding_model"] == "project-model" for hit in merged)
