from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.future import select

from app.db.models_sqlmodel import ChatTurn, Memory, MemoryEvent, Project, Thread
from app.models.requests import (
    MemoryCuratorApplyRequest,
    MemoryCuratorContext,
    MemoryCuratorMessage,
    MemoryCuratorOperation,
    MemoryCuratorRespondRequest,
    MemoryReviewCursor,
)
from app.services import memory_curator_service
from app.time_utils import iso_utc_z, utc_now


@pytest.fixture
def curator_sessionmaker(engine, monkeypatch):
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(memory_curator_service, "async_session_maker", maker)
    monkeypatch.setattr(
        memory_curator_service,
        "require_embedding_model_ready",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        memory_curator_service,
        "resolve_scope_embedding_model",
        AsyncMock(return_value="BAAI/bge-m3"),
    )
    monkeypatch.setattr(
        memory_curator_service,
        "index_memory_record",
        AsyncMock(return_value=1),
    )
    monkeypatch.setattr(
        memory_curator_service,
        "get_vector_db",
        lambda: SimpleNamespace(delete_memory_vectors=AsyncMock(return_value=True)),
    )
    return maker


async def _workspace(maker):
    async with maker() as session:
        async with session.begin():
            project = Project(
                id="curator-project",
                name="Curator",
                embedding_model="BAAI/bge-m3",
            )
            thread = Thread(
                id="curator-thread",
                project_id=project.id,
                name="Conversation",
                embedding_model=project.embedding_model,
            )
            session.add_all([project, thread])
    return project, thread


def _context():
    return MemoryCuratorContext(
        selected_scope_type="thread",
        selected_scope_id="curator-thread",
        thread_id="curator-thread",
        project_id="curator-project",
    )


@pytest.mark.asyncio
async def test_curator_confirmed_create_and_update_reuse_canonical_id(
    curator_sessionmaker,
):
    await _workspace(curator_sessionmaker)
    created = await memory_curator_service.apply_memory_curator_change_set(
        MemoryCuratorApplyRequest(
            context=_context(),
            confirmed=True,
            operations=[MemoryCuratorOperation(
                action="create",
                scope_type="thread",
                scope_id="curator-thread",
                memory_type="semantic",
                content="The preferred release channel is stable.",
            )],
        )
    )
    memory = created["changed_memories"][0]

    updated = await memory_curator_service.apply_memory_curator_change_set(
        MemoryCuratorApplyRequest(
            context=_context(),
            confirmed=True,
            operations=[MemoryCuratorOperation(
                action="update",
                scope_type="thread",
                scope_id="curator-thread",
                memory_id=memory["id"],
                expected_updated_at=memory["updated_at"],
                memory_type="semantic",
                content="The preferred release channel is preview.",
            )],
        )
    )

    assert updated["changed_memories"][0]["id"] == memory["id"]
    assert updated["changed_memories"][0]["content"].endswith("preview.")
    async with curator_sessionmaker() as session:
        rows = list((await session.execute(select(Memory))).scalars().all())
        events = list((await session.execute(
            select(MemoryEvent).where(MemoryEvent.memory_id == memory["id"])
        )).scalars().all())
    assert [row.id for row in rows] == [memory["id"]]
    assert [event.event_type for event in events] == ["curator_created", "curator_updated"]


@pytest.mark.asyncio
async def test_curator_rejects_stale_and_duplicate_change_sets(curator_sessionmaker):
    await _workspace(curator_sessionmaker)
    first = await memory_curator_service.apply_memory_curator_change_set(
        MemoryCuratorApplyRequest(
            context=_context(),
            confirmed=True,
            operations=[MemoryCuratorOperation(
                action="create",
                scope_type="thread",
                scope_id="curator-thread",
                memory_type="semantic",
                content="Use concise answers.",
            )],
        )
    )
    memory = first["changed_memories"][0]

    with pytest.raises(memory_curator_service.MemoryChangedError):
        await memory_curator_service.apply_memory_curator_change_set(
            MemoryCuratorApplyRequest(
                context=_context(),
                confirmed=True,
                operations=[MemoryCuratorOperation(
                    action="update",
                    scope_type="thread",
                    scope_id="curator-thread",
                    memory_id=memory["id"],
                    expected_updated_at="2000-01-01T00:00:00Z",
                    memory_type="semantic",
                    content="Use detailed answers.",
                )],
            )
        )

    with pytest.raises(memory_curator_service.MemoryCuratorError, match="identical"):
        await memory_curator_service.apply_memory_curator_change_set(
            MemoryCuratorApplyRequest(
                context=_context(),
                confirmed=True,
                operations=[MemoryCuratorOperation(
                    action="create",
                    scope_type="thread",
                    scope_id="curator-thread",
                    memory_type="semantic",
                    content="Use concise answers.",
                )],
            )
        )


@pytest.mark.asyncio
async def test_vector_cleanup_failure_preserves_canonical_memory(
    curator_sessionmaker,
    monkeypatch,
):
    await _workspace(curator_sessionmaker)
    first = await memory_curator_service.apply_memory_curator_change_set(
        MemoryCuratorApplyRequest(
            context=_context(),
            confirmed=True,
            operations=[MemoryCuratorOperation(
                action="create",
                scope_type="thread",
                scope_id="curator-thread",
                memory_type="semantic",
                content="Keep this original content.",
            )],
        )
    )
    memory = first["changed_memories"][0]
    monkeypatch.setattr(
        memory_curator_service,
        "get_vector_db",
        lambda: SimpleNamespace(delete_memory_vectors=AsyncMock(return_value=False)),
    )

    with pytest.raises(memory_curator_service.MemoryCuratorError, match="clean vectors"):
        await memory_curator_service.apply_memory_curator_change_set(
            MemoryCuratorApplyRequest(
                context=_context(),
                confirmed=True,
                operations=[MemoryCuratorOperation(
                    action="update",
                    scope_type="thread",
                    scope_id="curator-thread",
                    memory_id=memory["id"],
                    expected_updated_at=memory["updated_at"],
                    memory_type="semantic",
                    content="Do not persist this change.",
                )],
            )
        )

    async with curator_sessionmaker() as session:
        preserved = await session.get(Memory, memory["id"])
    assert preserved.content == "Keep this original content."


@pytest.mark.asyncio
async def test_no_change_confirmation_advances_review_cursor(curator_sessionmaker):
    _project, thread = await _workspace(curator_sessionmaker)
    now = utc_now()
    async with curator_sessionmaker() as session:
        async with session.begin():
            turn = ChatTurn(
                id="review-turn",
                thread_id=thread.id,
                status="completed",
                payload={"question": "Question", "answer": "Answer"},
                created_at=now,
                completed_at=now,
            )
            session.add(turn)

    result = await memory_curator_service.apply_memory_curator_change_set(
        MemoryCuratorApplyRequest(
            context=_context(),
            confirmed=True,
            operations=[],
            review_cursor=MemoryReviewCursor(
                thread_id=thread.id,
                reviewed_through_turn_id="review-turn",
                reviewed_through_created_at=now,
            ),
        )
    )

    assert result["review_cursor_advanced"] is True
    async with curator_sessionmaker() as session:
        refreshed = await session.get(Thread, thread.id)
    assert refreshed.thread_metadata["memory_curator"]["reviewed_through_turn_id"] == "review-turn"


@pytest.mark.asyncio
async def test_review_batches_latest_twenty_then_oldest_after_cursor(curator_sessionmaker):
    _project, thread = await _workspace(curator_sessionmaker)
    started = utc_now() - timedelta(days=2)
    async with curator_sessionmaker() as session:
        async with session.begin():
            for index in range(25):
                occurred_at = started + timedelta(minutes=index)
                session.add(ChatTurn(
                    id=f"initial-{index:02d}",
                    thread_id=thread.id,
                    status="completed",
                    payload={"question": f"Q{index}", "answer": f"A{index}"},
                    created_at=occurred_at,
                    completed_at=occurred_at,
                ))

    first = await memory_curator_service._review_batch(thread)
    assert first["reviewed_count"] == 20
    assert [turn["id"] for turn in first["turns"]] == [
        f"initial-{index:02d}" for index in range(5, 25)
    ]

    async with curator_sessionmaker() as session:
        async with session.begin():
            stored_thread = await session.get(Thread, thread.id)
            stored_thread.thread_metadata = {
                "memory_curator": {
                    "reviewed_through_turn_id": first["cursor"]["reviewed_through_turn_id"],
                    "reviewed_through_created_at": first["cursor"]["reviewed_through_created_at"],
                }
            }
            for index in range(25, 50):
                occurred_at = started + timedelta(minutes=index)
                session.add(ChatTurn(
                    id=f"later-{index:02d}",
                    thread_id=thread.id,
                    status="completed",
                    payload={"question": f"Q{index}", "answer": f"A{index}"},
                    created_at=occurred_at,
                    completed_at=occurred_at,
                ))
        await session.refresh(stored_thread)

    second = await memory_curator_service._review_batch(stored_thread)
    assert second["reviewed_count"] == 20
    assert second["remaining_count"] == 5
    assert [turn["id"] for turn in second["turns"]] == [
        f"later-{index:02d}" for index in range(25, 45)
    ]


@pytest.mark.asyncio
async def test_curator_malformed_model_output_becomes_clarification(
    curator_sessionmaker,
    monkeypatch,
):
    await _workspace(curator_sessionmaker)
    monkeypatch.setattr(
        memory_curator_service,
        "check_chat_model_ready",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        memory_curator_service,
        "_curator_memory_context",
        AsyncMock(return_value=([], [])),
    )
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="not json")))
    monkeypatch.setattr(memory_curator_service, "get_llm", lambda *_args, **_kwargs: llm)

    response = await memory_curator_service.respond_to_memory_curator(
        MemoryCuratorRespondRequest(
            mode="create",
            context=_context(),
            messages=[MemoryCuratorMessage(role="user", content="Remember something.")],
            llm_model="chat-model",
            context_window=8192,
        )
    )

    assert response["state"] == "clarification"
    assert response["operations"] == []
    assert response["choices"]
