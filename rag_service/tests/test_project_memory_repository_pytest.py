from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.enums import MemoryCandidateStatus, MemoryScopeType, MemoryStatus, MemoryType
from app.db.models_sqlmodel import Memory, MemoryCandidate, MemoryEvent, Thread
from app.db.repositories.memory_repo_sqlmodel import MemoryRepository
from app.db.repositories.project_repo_sqlmodel import DEFAULT_PROJECT_NAME, ProjectRepository
from app.db.repositories.thread_repo_sqlmodel import ThreadRepository
from app.time_utils import utc_now
from app.services.memory_promotion_service import extract_memory_candidates_from_text


@pytest.fixture
def repo_sessionmaker(engine, monkeypatch):
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    import app.db.repositories.memory_repo_sqlmodel as memory_repo
    import app.db.repositories.project_repo_sqlmodel as project_repo
    import app.db.repositories.thread_repo_sqlmodel as thread_repo

    monkeypatch.setattr(memory_repo, "async_session_maker", maker)
    monkeypatch.setattr(project_repo, "async_session_maker", maker)
    monkeypatch.setattr(thread_repo, "async_session_maker", maker)
    return maker


@pytest.mark.asyncio
async def test_default_project_backfills_orphan_threads(repo_sessionmaker):
    async with repo_sessionmaker() as session:
        async with session.begin():
            session.add(
                Thread(
                    id="orphan-thread",
                    name="Orphan",
                    embedding_model="BAAI/bge-m3",
                    settings={},
                    thread_metadata={},
                )
            )

    project = await ProjectRepository().ensure_default_project()

    async with repo_sessionmaker() as session:
        refreshed = await session.get(Thread, "orphan-thread")
    assert project.name == DEFAULT_PROJECT_NAME
    assert refreshed.project_id == project.id


@pytest.mark.asyncio
async def test_thread_repository_creates_thread_in_project(repo_sessionmaker):
    project = await ProjectRepository().create(name="Research")

    thread = await ThreadRepository().create(
        "Thread",
        "BAAI/bge-m3",
        project_id=project.id,
    )

    assert thread.project_id == project.id


@pytest.mark.asyncio
async def test_memory_repository_lifecycle_and_audit(repo_sessionmaker):
    repo = MemoryRepository()

    memory = await repo.create_memory(
        scope_type=MemoryScopeType.PROJECT.value,
        scope_id="project-1",
        memory_type=MemoryType.SEMANTIC.value,
        content="The project codename is Atlas.",
        confidence=0.9,
        created_by="test",
    )
    archived = await repo.update_memory_status(
        memory.id,
        status=MemoryStatus.ARCHIVED.value,
        actor_id="reviewer",
    )

    assert archived.status == MemoryStatus.ARCHIVED.value
    rows = await repo.list_memories(
        scope_type=MemoryScopeType.PROJECT.value,
        scope_id="project-1",
        status=MemoryStatus.ARCHIVED.value,
    )
    assert [row.id for row in rows] == [memory.id]
    async with repo_sessionmaker() as session:
        event_result = await session.execute(select(MemoryEvent).where(MemoryEvent.memory_id == memory.id))
        assert len(list(event_result.scalars().all())) == 2


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
    )
    expired = await repo.create_memory(
        scope_type=MemoryScopeType.PROJECT.value,
        scope_id="project-cleanup",
        memory_type=MemoryType.SEMANTIC.value,
        content="Expired cleanup memory.",
        expires_at=now - timedelta(minutes=1),
    )
    fresh = await repo.create_memory(
        scope_type=MemoryScopeType.PROJECT.value,
        scope_id="project-cleanup",
        memory_type=MemoryType.SEMANTIC.value,
        content="Fresh memory.",
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


@pytest.mark.asyncio
async def test_memory_candidate_resolution(repo_sessionmaker):
    repo = MemoryRepository()

    candidate = await repo.create_candidate(
        proposed_scope_type=MemoryScopeType.THREAD.value,
        proposed_scope_id="thread-1",
        memory_type=MemoryType.SEMANTIC.value,
        content="Use concise answers.",
    )
    resolved = await repo.resolve_candidate(
        candidate.id,
        status=MemoryCandidateStatus.REJECTED.value,
    )

    assert resolved.status == MemoryCandidateStatus.REJECTED.value


@pytest.mark.asyncio
async def test_memory_repository_hard_deletes_candidates(repo_sessionmaker):
    repo = MemoryRepository()

    direct = await repo.create_candidate(
        proposed_scope_type=MemoryScopeType.PROJECT.value,
        proposed_scope_id="project-1",
        memory_type=MemoryType.SEMANTIC.value,
        content="Candidate to delete.",
    )
    thread_source = await repo.create_candidate(
        proposed_scope_type=MemoryScopeType.PROJECT.value,
        proposed_scope_id="project-1",
        source_thread_id="thread-1",
        memory_type=MemoryType.SEMANTIC.value,
        content="Thread sourced candidate.",
    )
    thread_scope = await repo.create_candidate(
        proposed_scope_type=MemoryScopeType.THREAD.value,
        proposed_scope_id="thread-1",
        memory_type=MemoryType.SEMANTIC.value,
        content="Thread scoped candidate.",
    )

    assert await repo.delete_candidate(direct.id) is True
    deleted_thread_candidates = await repo.delete_candidates_for_thread("thread-1")

    assert set(deleted_thread_candidates) == {thread_source.id, thread_scope.id}
    async with repo_sessionmaker() as session:
        remaining = (await session.execute(select(MemoryCandidate))).scalars().all()
        assert remaining == []


@pytest.mark.asyncio
async def test_memory_repository_rejects_invalid_candidate_values(repo_sessionmaker):
    repo = MemoryRepository()

    with pytest.raises(ValueError):
        await repo.create_candidate(
            proposed_scope_type="workspace",
            proposed_scope_id="thread-1",
            memory_type=MemoryType.SEMANTIC.value,
            content="content",
        )
    with pytest.raises(ValueError):
        await repo.create_candidate(
            proposed_scope_type=MemoryScopeType.THREAD.value,
            proposed_scope_id="thread-1",
            memory_type="fact",
            content="content",
        )
    with pytest.raises(ValueError):
        await repo.create_candidate(
            proposed_scope_type=MemoryScopeType.THREAD.value,
            proposed_scope_id="thread-1",
            memory_type=MemoryType.SEMANTIC.value,
            content="",
        )
    with pytest.raises(ValueError):
        await repo.resolve_candidate("missing", status="done")


def test_explicit_remember_phrase_extracts_project_candidate():
    proposals = extract_memory_candidates_from_text(
        "Remember for this project that the product name is AskPDF Pro.",
        thread_id="thread-1",
        project_id="project-1",
    )

    assert len(proposals) == 1
    assert proposals[0].scope_type == MemoryScopeType.PROJECT.value
    assert proposals[0].scope_id == "project-1"
    assert proposals[0].content == "the product name is AskPDF Pro."


def test_explicit_user_memory_extracts_user_candidate():
    proposals = extract_memory_candidates_from_text(
        "Please remember for me that I prefer concise answers.",
        thread_id="thread-1",
        project_id="project-1",
        user_id="user-1",
    )

    assert len(proposals) == 1
    assert proposals[0].scope_type == MemoryScopeType.USER.value
    assert proposals[0].scope_id == "user-1"
