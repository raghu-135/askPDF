"""
test_message_repository_pytest.py - ChatTurn-backed message compatibility tests.
"""

import os
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import select


from app.db.models_sqlmodel import ChatTurn, MessageRole
from app.db.repositories.message_repo_sqlmodel import (
    MessageRepository,
    message_id_for_turn,
)

SQLMODEL_AVAILABLE = bool(os.getenv("TEST_DATABASE_URL"))


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel test database is not configured")
class TestMessageRepository:
    @pytest_asyncio.fixture
    async def repo(self, engine):
        session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        async with session_factory() as repo_session:
            yield MessageRepository(repo_session)

    @pytest.mark.asyncio
    async def test_create_turn_expands_to_user_and_assistant_messages(self, repo, sample_thread):
        created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        turn = await repo.create_turn(
            thread_id=sample_thread.id,
            question="What changed?",
            rewritten_question="Summarize the change",
            answer="We now store chat turns.",
            reasoning="Paired rows became one row.",
            reasoning_available=True,
            reasoning_format="markdown",
            web_sources=[{"url": "https://example.com"}],
            metadata={"context_compact": "Q/A compact"},
            created_at=created_at,
        )

        messages = await repo.get_thread_messages(sample_thread.id)

        assert [m.role for m in messages] == [MessageRole.USER.value, MessageRole.ASSISTANT.value]
        assert messages[0].id == message_id_for_turn(turn.id, MessageRole.USER.value)
        assert messages[0].content == "What changed?"
        assert messages[0].context_compact == "Summarize the change"
        assert messages[1].id == message_id_for_turn(turn.id, MessageRole.ASSISTANT.value)
        assert messages[1].content == "We now store chat turns."
        assert messages[1].reasoning_available is True
        assert messages[1].web_sources == [{"url": "https://example.com"}]

    @pytest.mark.asyncio
    async def test_get_message_by_compatibility_id(self, repo, sample_thread):
        turn = await repo.create_turn(
            thread_id=sample_thread.id,
            question="Question",
            answer="Answer",
        )

        user = await repo.get(f"{turn.id}:user")
        assistant = await repo.get(f"{turn.id}:assistant")

        assert user.role == MessageRole.USER.value
        assert user.content == "Question"
        assert assistant.role == MessageRole.ASSISTANT.value
        assert assistant.content == "Answer"

    @pytest.mark.asyncio
    async def test_recent_messages_expand_recent_turns_in_chronological_order(self, repo, sample_thread):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(3):
            await repo.create_turn(
                thread_id=sample_thread.id,
                question=f"Q{i}",
                answer=f"A{i}",
                created_at=base + timedelta(minutes=i),
            )

        messages = await repo.get_recent_messages(sample_thread.id, limit=2)

        assert [m.content for m in messages] == ["Q2", "A2"]

    @pytest.mark.asyncio
    async def test_thread_message_pagination_applies_to_expanded_messages(self, repo, sample_thread):
        await repo.create_turn(
            thread_id=sample_thread.id,
            question="Q0",
            answer="A0",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        await repo.create_turn(
            thread_id=sample_thread.id,
            question="Q1",
            answer="A1",
            created_at=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
        )

        first = await repo.get_thread_messages(sample_thread.id, limit=1, offset=0)
        second = await repo.get_thread_messages(sample_thread.id, limit=1, offset=1)
        middle = await repo.get_thread_messages(sample_thread.id, limit=2, offset=1)

        assert [m.content for m in first] == ["Q0"]
        assert [m.content for m in second] == ["A0"]
        assert [m.content for m in middle] == ["A0", "Q1"]

    @pytest.mark.asyncio
    async def test_update_context_reasoning_and_sources_mutates_payload(self, repo, session, sample_thread):
        turn = await repo.create_turn(
            thread_id=sample_thread.id,
            question="Question",
            answer="Answer",
        )

        assert await repo.update_context_compact(f"{turn.id}:assistant", "Compact memory")
        assert await repo.update_reasoning(turn.id, "Reasoning text", reasoning_format="markdown")
        assert await repo.update_web_sources(turn.id, [{"url": "https://example.com"}])

        refreshed = (
            await session.execute(select(ChatTurn).where(ChatTurn.id == turn.id))
        ).scalar_one()
        assert refreshed.payload["metadata"]["context_compact"] == "Compact memory"
        assert refreshed.payload["reasoning"] == "Reasoning text"
        assert refreshed.payload["reasoning_format"] == "markdown"
        assert refreshed.payload["web_sources"] == [{"url": "https://example.com"}]

    @pytest.mark.asyncio
    async def test_delete_pair_deletes_owning_turn_from_either_compatibility_id(self, repo, session, sample_thread):
        turn = await repo.create_turn(
            thread_id=sample_thread.id,
            question="Question",
            answer="Answer",
        )

        deleted_ids = await repo.delete_pair(f"{turn.id}:user")

        assert deleted_ids == [f"{turn.id}:user", f"{turn.id}:assistant"]
        remaining = (
            await session.execute(select(ChatTurn).where(ChatTurn.id == turn.id))
        ).scalar_one_or_none()
        assert remaining is None

    @pytest.mark.asyncio
    async def test_cancelled_turns_are_hidden(self, repo, session, sample_thread):
        session.add(
            ChatTurn(
                id="cancelled-turn",
                thread_id=sample_thread.id,
                status="cancelled",
                payload={"question": "Hidden", "answer": "Also hidden"},
            )
        )
        await session.commit()

        assert await repo.get("cancelled-turn:user") is None
        assert await repo.get_thread_messages(sample_thread.id) == []
