from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.future import select


from app.db.models_sqlmodel import (
    ChatTurn,
    File,
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
    assert forked.settings == {"max_iterations": 3}
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
