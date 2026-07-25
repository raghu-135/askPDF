from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.future import select


from app.db.models_sqlmodel import (
    ChatTurn,
    File,
    Memory,
    MemoryEvent,
    Project,
    Thread,
    ThreadFile,
)
from app.services import thread_management_service


@pytest.mark.asyncio
async def test_fork_thread_from_message_copies_lineage_and_prior_rows(engine, monkeypatch):
    test_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    monkeypatch.setattr(
        thread_management_service,
        "async_session_maker",
        test_session_maker,
    )

    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    async with test_session_maker() as session:
        async with session.begin():
            session.add_all(
                [
                Project(id="project-a", name="Project A", embedding_model="BAAI/bge-m3"),
                Thread(
                    id="source-thread",
                    project_id="project-a",
                    name="Source Thread",
                    embedding_model="BAAI/bge-m3",
                    settings={"replans": 3},
                    thread_metadata={"existing": True},
                    created_at=created_at,
                ),
                ]
            )
            session.add(
                File(
                    file_hash="file-1",
                    file_name="paper.pdf",
                    file_path="/data/paper.pdf",
                    source_type="pdf",
                )
            )
            await session.flush()
            session.add(
                ThreadFile(
                    thread_id="source-thread",
                    file_hash="file-1",
                    added_at=created_at,
                    annotations=[{"id": "a1"}],
                    annotations_updated_at=created_at,
                )
            )
            session.add_all(
                [
                    ChatTurn(
                        id="turn-1",
                        thread_id="source-thread",
                        status="completed",
                        payload={
                            "question": "question",
                            "rewritten_question": None,
                            "answer": "answer",
                            "reasoning": "",
                            "reasoning_available": False,
                            "reasoning_format": "none",
                            "web_sources": [{"url": "https://example.com"}],
                            "document_sources": [],
                            "used_chat_ids": [],
                            "clarification_options": None,
                            "error": None,
                            "metadata": {"context_compact": "Q: question\nA: answer"},
                        },
                        created_at=created_at,
                        completed_at=created_at,
                    ),
                    ChatTurn(
                        id="turn-2",
                        thread_id="source-thread",
                        status="failed",
                        payload={
                            "question": "later question",
                            "rewritten_question": None,
                            "answer": None,
                            "reasoning": "",
                            "reasoning_available": False,
                            "reasoning_format": "none",
                            "web_sources": [],
                            "document_sources": [],
                            "used_chat_ids": [],
                            "clarification_options": None,
                            "error": {"code": "missing_assistant_message"},
                            "metadata": {},
                        },
                        created_at=created_at,
                    ),
                ]
            )

    result = await thread_management_service.fork_thread(
        "source-thread",
        message_id="turn-1:assistant",
        name="Forked Thread",
    )
    forked = result["thread"]

    async with test_session_maker() as session:
        turns = (
            await session.execute(
                select(ChatTurn)
                .where(ChatTurn.thread_id == forked.id)
                .order_by(ChatTurn.created_at.asc())
            )
        ).scalars().all()
        files = (
            await session.execute(
                select(ThreadFile).where(ThreadFile.thread_id == forked.id)
            )
        ).scalars().all()
        source_thread = (
            await session.execute(
                select(Thread).where(Thread.id == "source-thread")
            )
        ).scalar_one()

    assert forked.name == "Forked Thread"
    assert forked.settings == {"replans": 3}
    assert forked.thread_metadata["existing"] is True
    assert "fork_children" not in forked.thread_metadata
    assert forked.thread_metadata["fork"]["parent_thread_id"] == "source-thread"
    assert forked.thread_metadata["fork"]["parent_thread_name"] == "Source Thread"
    assert forked.thread_metadata["fork"]["source_message_id"] == "turn-1:assistant"
    assert forked.thread_metadata["fork"]["mode"] == "from_message"
    assert source_thread.thread_metadata["fork_children"] == [forked.id]
    assert [t.payload["question"] for t in turns] == ["question"]
    assert [t.payload["answer"] for t in turns] == ["answer"]
    assert all(t.id not in {"turn-1", "turn-2"} for t in turns)
    assert [f.file_hash for f in files] == ["file-1"]
    assert files[0].annotations == [{"id": "a1"}]
    assert files[0].annotations_updated_at == created_at


@pytest.mark.asyncio
async def test_fork_thread_rejects_message_from_another_thread(engine, monkeypatch):
    test_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    monkeypatch.setattr(
        thread_management_service,
        "async_session_maker",
        test_session_maker,
    )

    async with test_session_maker() as session:
        async with session.begin():
            session.add_all(
                [
                    Project(id="project-a", name="Project A", embedding_model="BAAI/bge-m3"),
                    Thread(
                        id="source-thread",
                        project_id="project-a",
                        name="Source Thread",
                        embedding_model="BAAI/bge-m3",
                        settings={},
                        thread_metadata={},
                    ),
                    Thread(
                        id="other-thread",
                        project_id="project-a",
                        name="Other Thread",
                        embedding_model="BAAI/bge-m3",
                        settings={},
                        thread_metadata={},
                    ),
                ]
            )
            await session.flush()
            session.add(
                ChatTurn(
                    id="other-turn",
                    thread_id="other-thread",
                    status="completed",
                    payload={"question": "wrong thread", "answer": "wrong answer"},
                )
            )

    with pytest.raises(thread_management_service.ForkMessageNotFoundError):
        await thread_management_service.fork_thread(
            "source-thread",
            message_id="other-turn:user",
        )


@pytest.mark.asyncio
async def test_same_project_fork_snapshots_thread_memories_before_message(engine, monkeypatch):
    test_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    monkeypatch.setattr(
        thread_management_service,
        "async_session_maker",
        test_session_maker,
    )
    indexed_memory_ids: list[str] = []

    async def fake_index_memory_record(memory):
        indexed_memory_ids.append(memory.id)
        return 1

    monkeypatch.setattr("app.services.memory_service.index_memory_record", fake_index_memory_record)

    early_at = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    fork_at = datetime(2026, 1, 1, 11, tzinfo=timezone.utc)
    late_at = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    async with test_session_maker() as session:
        async with session.begin():
            session.add(Project(id="project-a", name="Project A", embedding_model="BAAI/bge-m3"))
            session.add(
                Thread(
                    id="source-thread",
                    project_id="project-a",
                    name="Source Thread",
                    embedding_model="BAAI/bge-m3",
                    settings={},
                    thread_metadata={},
                    created_at=early_at,
                )
            )
            session.add_all(
                [
                    ChatTurn(
                        id="turn-1",
                        thread_id="source-thread",
                        status="completed",
                        payload={"question": "q1", "answer": "a1"},
                        created_at=fork_at,
                        completed_at=fork_at,
                    ),
                    ChatTurn(
                        id="turn-2",
                        thread_id="source-thread",
                        status="completed",
                        payload={"question": "q2", "answer": "a2"},
                        created_at=late_at,
                        completed_at=late_at,
                    ),
                    Memory(
                        id="memory-before",
                        scope_type="thread",
                        scope_id="source-thread",
                        memory_type="semantic",
                        content="Remembered before fork",
                        embedding_model="BAAI/bge-m3",
                        content_hash="memory-before-hash",
                        summary="before",
                        source_refs_json={"turn_id": "turn-1"},
                        confidence=0.9,
                        status="active",
                        visibility="private",
                        created_at=early_at,
                        updated_at=early_at,
                    ),
                    Memory(
                        id="memory-after",
                        scope_type="thread",
                        scope_id="source-thread",
                        memory_type="semantic",
                        content="Remembered after fork",
                        embedding_model="BAAI/bge-m3",
                        content_hash="memory-after-hash",
                        summary="after",
                        source_refs_json={"turn_id": "turn-2"},
                        confidence=0.9,
                        status="active",
                        visibility="private",
                        created_at=late_at,
                        updated_at=late_at,
                    ),
                    Memory(
                        id="project-memory",
                        scope_type="project",
                        scope_id="project-a",
                        memory_type="semantic",
                        content="Shared project memory stays shared",
                        embedding_model="BAAI/bge-m3",
                        content_hash="project-memory-hash",
                        summary="project",
                        confidence=0.9,
                        status="active",
                        visibility="project",
                        created_at=early_at,
                        updated_at=early_at,
                    ),
                ]
            )

    result = await thread_management_service.fork_thread(
        "source-thread",
        message_id="turn-1:assistant",
    )
    forked = result["thread"]

    async with test_session_maker() as session:
        copied_memories = (
            await session.execute(
                select(Memory)
                .where(Memory.scope_id == forked.id)
                .order_by(Memory.created_at.asc())
            )
        ).scalars().all()
        snapshot_events = (
            await session.execute(
                select(MemoryEvent).where(MemoryEvent.event_type == "fork_snapshot")
            )
        ).scalars().all()

    assert forked.project_id == "project-a"
    assert forked.thread_metadata["fork"]["memory_copy_mode"] == "thread_snapshot"
    assert len(copied_memories) == 1
    copied = copied_memories[0]
    assert copied.content == "Remembered before fork"
    assert copied.scope_type == "thread"
    assert copied.scope_id == forked.id
    assert copied.fork_origin_json["source_memory_id"] == "memory-before"
    assert copied.fork_origin_json["copy_mode"] == "thread_snapshot"
    assert forked.thread_metadata["fork"]["copied_memory_ids"] == [copied.id]
    assert [event.memory_id for event in snapshot_events] == [copied.id]
    assert indexed_memory_ids == [copied.id]


@pytest.mark.asyncio
async def test_new_project_fork_snapshots_project_memory_and_diverges(engine, monkeypatch):
    test_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    monkeypatch.setattr(
        thread_management_service,
        "async_session_maker",
        test_session_maker,
    )
    indexed_memory_ids: list[str] = []

    async def fake_index_memory_record(memory):
        indexed_memory_ids.append(memory.id)
        return 1

    monkeypatch.setattr("app.services.memory_service.index_memory_record", fake_index_memory_record)

    created_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    async with test_session_maker() as session:
        async with session.begin():
            session.add_all(
                [
                    Project(id="project-a", name="Project A", embedding_model="BAAI/bge-m3"),
                    Project(id="project-b", name="Project B", embedding_model="BAAI/bge-m3"),
                    Thread(
                        id="source-thread",
                        project_id="project-a",
                        name="Source Thread",
                        embedding_model="BAAI/bge-m3",
                        settings={},
                        thread_metadata={},
                        created_at=created_at,
                    ),
                    ChatTurn(
                        id="turn-1",
                        thread_id="source-thread",
                        status="completed",
                        payload={"question": "q1", "answer": "a1"},
                        created_at=created_at,
                        completed_at=created_at,
                    ),
                    Memory(
                        id="source-project-memory",
                        scope_type="project",
                        scope_id="project-a",
                        memory_type="semantic",
                        content="Project A launch name is AskPDF Pro",
                        embedding_model="BAAI/bge-m3",
                        content_hash="source-project-memory-hash",
                        summary="launch name",
                        source_refs_json={"project": "a"},
                        confidence=0.95,
                        status="active",
                        visibility="project",
                        created_at=created_at,
                        updated_at=created_at,
                    ),
                    Memory(
                        id="source-thread-memory",
                        scope_type="thread",
                        scope_id="source-thread",
                        memory_type="semantic",
                        content="Thread-local detail",
                        embedding_model="BAAI/bge-m3",
                        content_hash="source-thread-memory-hash",
                        summary="thread detail",
                        confidence=0.8,
                        status="active",
                        visibility="private",
                        created_at=created_at,
                        updated_at=created_at,
                    ),
                ]
            )

    result = await thread_management_service.fork_thread(
        "source-thread",
        target_project_id="project-b",
    )
    forked = result["thread"]

    async with test_session_maker() as session:
        project_b_memories = (
            await session.execute(
                select(Memory).where(Memory.scope_type == "project", Memory.scope_id == "project-b")
            )
        ).scalars().all()
        fork_thread_memories = (
            await session.execute(
                select(Memory).where(Memory.scope_type == "thread", Memory.scope_id == forked.id)
            )
        ).scalars().all()

    assert forked.project_id == "project-b"
    assert forked.thread_metadata["fork"]["memory_copy_mode"] == "project_snapshot"
    assert fork_thread_memories == []
    assert len(project_b_memories) == 1
    copied = project_b_memories[0]
    assert copied.content == "Project A launch name is AskPDF Pro"
    assert copied.fork_origin_json["source_memory_id"] == "source-project-memory"
    assert copied.fork_origin_json["source_project_id"] == "project-a"
    assert copied.fork_origin_json["target_project_id"] == "project-b"
    assert copied.fork_origin_json["copy_mode"] == "project_snapshot"
    assert forked.thread_metadata["fork"]["copied_memory_ids"] == [copied.id]
    assert indexed_memory_ids == [copied.id]
