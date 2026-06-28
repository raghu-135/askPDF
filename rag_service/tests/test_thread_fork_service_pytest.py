import os
import sys
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.future import select

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.db.models_sqlmodel import (
    File,
    Message,
    Thread,
    ThreadFile,
    ThreadFileAnnotation,
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
            session.add(
                Thread(
                    id="source-thread",
                    name="Source Thread",
                    embed_model="BAAI/bge-m3",
                    settings={"max_iterations": 3},
                    thread_metadata={"existing": True},
                    created_at=created_at,
                )
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
                )
            )
            session.add(
                ThreadFileAnnotation(
                    thread_id="source-thread",
                    file_hash="file-1",
                    annotations_json='[{"id":"a1"}]',
                    created_at=created_at,
                    updated_at=created_at,
                )
            )
            session.add_all(
                [
                    Message(
                        id="m1",
                        thread_id="source-thread",
                        role="user",
                        content="question",
                        created_at=created_at,
                    ),
                    Message(
                        id="m2",
                        thread_id="source-thread",
                        role="assistant",
                        content="answer",
                        context_compact="Q: question\nA: answer",
                        reasoning_available=False,
                        reasoning_format="none",
                        web_sources=[{"url": "https://example.com"}],
                        created_at=created_at,
                    ),
                    Message(
                        id="m3",
                        thread_id="source-thread",
                        role="user",
                        content="later question",
                        created_at=created_at,
                    ),
                ]
            )

    result = await thread_management_service.fork_thread(
        "source-thread",
        message_id="m2",
        name="Forked Thread",
    )
    forked = result["thread"]

    async with test_session_maker() as session:
        messages = (
            await session.execute(
                select(Message)
                .where(Message.thread_id == forked.id)
                .order_by(Message.created_at.asc())
            )
        ).scalars().all()
        files = (
            await session.execute(
                select(ThreadFile).where(ThreadFile.thread_id == forked.id)
            )
        ).scalars().all()
        annotations = (
            await session.execute(
                select(ThreadFileAnnotation).where(
                    ThreadFileAnnotation.thread_id == forked.id
                )
            )
        ).scalars().all()
        source_thread = (
            await session.execute(
                select(Thread).where(Thread.id == "source-thread")
            )
        ).scalar_one()

    assert forked.name == "Forked Thread"
    assert forked.settings == {"max_iterations": 3}
    assert forked.thread_metadata["existing"] is True
    assert "fork_children" not in forked.thread_metadata
    assert forked.thread_metadata["fork"]["parent_thread_id"] == "source-thread"
    assert forked.thread_metadata["fork"]["parent_thread_name"] == "Source Thread"
    assert forked.thread_metadata["fork"]["source_message_id"] == "m2"
    assert forked.thread_metadata["fork"]["mode"] == "from_message"
    assert source_thread.thread_metadata["fork_children"] == [forked.id]
    assert [m.content for m in messages] == ["question", "answer"]
    assert all(m.id not in {"m1", "m2", "m3"} for m in messages)
    assert [f.file_hash for f in files] == ["file-1"]
    assert annotations[0].annotations_json == '[{"id":"a1"}]'


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
                    Thread(
                        id="source-thread",
                        name="Source Thread",
                        embed_model="BAAI/bge-m3",
                        settings={},
                        thread_metadata={},
                    ),
                    Thread(
                        id="other-thread",
                        name="Other Thread",
                        embed_model="BAAI/bge-m3",
                        settings={},
                        thread_metadata={},
                    ),
                ]
            )
            await session.flush()
            session.add(
                Message(
                    id="other-message",
                    thread_id="other-thread",
                    role="user",
                    content="wrong thread",
                )
            )

    with pytest.raises(thread_management_service.ForkMessageNotFoundError):
        await thread_management_service.fork_thread(
            "source-thread",
            message_id="other-message",
        )
