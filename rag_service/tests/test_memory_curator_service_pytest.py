import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.future import select

from app.db.models_sqlmodel import ChatTurn, Memory, MemoryEvent, MemoryOverride, Project, Thread
from app.models.requests import (
    MemoryCuratorApplyRequest,
    MemoryCuratorContext,
    MemoryCuratorMessage,
    MemoryCuratorOperation,
    MemoryCuratorRespondRequest,
    MemoryReviewCursor,
)
from app.models.memory_tools import (
    MEMORY_PROPOSE,
    MEMORY_READ_STORED,
    MemoryGetInput,
    MemoryPrepareChangeInput,
    MemorySearchInput,
)
from app.services import memory_curator_service, memory_tool_service
from app.time_utils import iso_utc_z, utc_now


@pytest.fixture
def curator_sessionmaker(engine, monkeypatch):
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    import app.services.effective_memory_service as effective_memory_service
    import app.services.memory_tool_service as memory_tool_service

    monkeypatch.setattr(memory_curator_service, "async_session_maker", maker)
    monkeypatch.setattr(effective_memory_service, "async_session_maker", maker)
    monkeypatch.setattr(memory_tool_service, "async_session_maker", maker)
    monkeypatch.setattr(
        memory_curator_service,
        "check_model_supports_tools",
        AsyncMock(return_value=False),
    )
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
    monkeypatch.setattr(
        memory_curator_service,
        "search_memory_tool",
        AsyncMock(return_value={"memories": [], "readiness": [], "truncated": False}),
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


def test_permission_only_choice_detection_preserves_real_conflict_options():
    assert memory_curator_service._permission_only_choices([
        {"label": "Yes, update it", "user_message": "Yes, update it."},
        {"label": "Cancel", "user_message": "No, cancel."},
    ]) is True
    assert memory_curator_service._permission_only_choices([
        {"label": "Update Project", "user_message": "Update the project memory."},
        {"label": "Override in Thread", "user_message": "Keep the project memory and override it here."},
    ]) is False


def test_curator_payload_hides_scope_ids_but_preserves_memory_ids():
    payload = memory_curator_service._curator_safe_payload({
        "id": "memory-1",
        "scope_type": "project",
        "scope_id": "project-1",
        "overrides": [{"id": "memory-2", "scope_id": "default"}],
    })

    assert payload["id"] == "memory-1"
    assert payload["overrides"][0]["id"] == "memory-2"
    assert "scope_id" not in payload
    assert "scope_id" not in payload["overrides"][0]


def test_curator_uses_typed_choice_for_cross_scope_override_resolution():
    assert memory_curator_service._selected_override_resolution([{
        "role": "user",
        "content": "For this thread, focus on NVIDIA and its AI systems.",
    }]) is None
    assert memory_curator_service._selected_override_resolution([{
        "role": "user",
        "content": "Keep both.",
        "choice_id": "keep-both",
    }]) == "additive"
    assert memory_curator_service._selected_override_resolution([{
        "role": "user",
        "content": "Use the narrower memory here.",
        "choice_id": "override-broader",
    }]) == "override"


def test_memory_curator_prompt_contains_relationship_examples():
    prompt = memory_curator_service.load_prompt("memory_curator/system.md")

    assert "Similarity is not conflict" in prompt
    assert 'id="keep-both"' in prompt
    assert 'id="override-broader"' in prompt
    assert "Same-scope correction updates the existing record" in prompt


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
    assert [event.event_type for event in events] == [
        "curator_created",
        "override_set",
        "curator_updated",
        "override_set",
    ]


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
                    content="Use concise answers.",
                )],
            )
        )


@pytest.mark.asyncio
async def test_relation_only_update_replaces_edges_without_reindexing(
    curator_sessionmaker,
    monkeypatch,
):
    await _workspace(curator_sessionmaker)
    global_result = await memory_curator_service.apply_memory_curator_change_set(
        MemoryCuratorApplyRequest(
            context=_context(),
            confirmed=True,
            operations=[MemoryCuratorOperation(
                action="create",
                scope_type="user",
                scope_id="default",
                content="Use concise answers everywhere.",
            )],
        )
    )
    thread_result = await memory_curator_service.apply_memory_curator_change_set(
        MemoryCuratorApplyRequest(
            context=_context(),
            confirmed=True,
            operations=[MemoryCuratorOperation(
                action="create",
                scope_type="thread",
                scope_id="curator-thread",
                content="Use detailed answers here.",
            )],
        )
    )
    global_memory = global_result["changed_memories"][0]
    thread_memory = thread_result["changed_memories"][0]
    index_mock = AsyncMock(return_value=1)
    delete_vectors = AsyncMock(return_value=True)
    monkeypatch.setattr(memory_curator_service, "index_memory_record", index_mock)
    monkeypatch.setattr(
        memory_curator_service,
        "get_vector_db",
        lambda: SimpleNamespace(delete_memory_vectors=delete_vectors),
    )

    related = await memory_curator_service.apply_memory_curator_change_set(
        MemoryCuratorApplyRequest(
            context=_context(),
            confirmed=True,
            operations=[MemoryCuratorOperation(
                action="update",
                scope_type="thread",
                scope_id="curator-thread",
                memory_id=thread_memory["id"],
                expected_updated_at=thread_memory["updated_at"],
                content=thread_memory["content"],
                override_targets=[{
                    "memory_id": global_memory["id"],
                    "expected_updated_at": global_memory["updated_at"],
                }],
            )],
        )
    )

    assert related["changed_memories"][0]["overrides"][0]["id"] == global_memory["id"]
    assert related["changed_memories"][0]["updated_at"] != thread_memory["updated_at"]
    index_mock.assert_not_awaited()
    delete_vectors.assert_not_awaited()


@pytest.mark.asyncio
async def test_curator_rejects_duplicate_targets_and_override_cycles(curator_sessionmaker):
    await _workspace(curator_sessionmaker)
    created = []
    for content in ("Preference A.", "Preference B."):
        result = await memory_curator_service.apply_memory_curator_change_set(
            MemoryCuratorApplyRequest(
                context=_context(),
                confirmed=True,
                operations=[MemoryCuratorOperation(
                    action="create",
                    scope_type="thread",
                    scope_id="curator-thread",
                    content=content,
                )],
            )
        )
        created.append(result["changed_memories"][0])
    first, second = created
    target = {
        "memory_id": second["id"],
        "expected_updated_at": second["updated_at"],
    }
    with pytest.raises(memory_curator_service.MemoryCuratorError, match="duplicates"):
        await memory_curator_service.apply_memory_curator_change_set(
            MemoryCuratorApplyRequest(
                context=_context(),
                confirmed=True,
                operations=[MemoryCuratorOperation(
                    action="update",
                    scope_type="thread",
                    scope_id="curator-thread",
                    memory_id=first["id"],
                    expected_updated_at=first["updated_at"],
                    content=first["content"],
                    override_targets=[target, target],
                )],
            )
        )

    first_updated = await memory_curator_service.apply_memory_curator_change_set(
        MemoryCuratorApplyRequest(
            context=_context(),
            confirmed=True,
            operations=[MemoryCuratorOperation(
                action="update",
                scope_type="thread",
                scope_id="curator-thread",
                memory_id=first["id"],
                expected_updated_at=first["updated_at"],
                content=first["content"],
                override_targets=[target],
            )],
        )
    )
    first = first_updated["changed_memories"][0]
    with pytest.raises(memory_curator_service.MemoryCuratorError, match="cycle"):
        await memory_curator_service.apply_memory_curator_change_set(
            MemoryCuratorApplyRequest(
                context=_context(),
                confirmed=True,
                operations=[MemoryCuratorOperation(
                    action="update",
                    scope_type="thread",
                    scope_id="curator-thread",
                    memory_id=second["id"],
                    expected_updated_at=second["updated_at"],
                    content=second["content"],
                    override_targets=[{
                        "memory_id": first["id"],
                        "expected_updated_at": first["updated_at"],
                    }],
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


@pytest.mark.asyncio
async def test_curator_complete_operation_skips_redundant_permission_step(
    curator_sessionmaker,
    monkeypatch,
):
    await _workspace(curator_sessionmaker)
    monkeypatch.setattr(memory_curator_service, "check_chat_model_ready", AsyncMock(return_value=True))
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=json.dumps({
        "message": "I can save that preference.",
        "state": "clarification",
        "choices": [{
            "id": "yes",
            "label": "Yes, save it",
            "description": "",
            "user_message": "Yes, save it.",
        }],
        "operations": [{
            "action": "create",
            "scope_type": "thread",
            "scope_id": "curator-thread",
            "content": "Answer in a funny way.",
            "override_targets": [],
        }],
    }))))
    monkeypatch.setattr(memory_curator_service, "get_llm", lambda *_args, **_kwargs: llm)

    response = await memory_curator_service.respond_to_memory_curator(
        MemoryCuratorRespondRequest(
            mode="create",
            context=_context(),
            messages=[MemoryCuratorMessage(role="user", content="Answer in a funny way.")],
            llm_model="chat-model",
            context_window=8192,
        )
    )

    assert response["state"] == "proposal"
    assert response["choices"] == []
    assert response["operations"][0]["content"] == "Answer in a funny way."
    assert llm.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_curator_repairs_permission_only_response_without_another_user_turn(
    curator_sessionmaker,
    monkeypatch,
):
    await _workspace(curator_sessionmaker)
    monkeypatch.setattr(memory_curator_service, "check_chat_model_ready", AsyncMock(return_value=True))
    permission = SimpleNamespace(content=json.dumps({
        "message": "Would you like me to save this?",
        "state": "clarification",
        "choices": [{
            "id": "yes",
            "label": "Yes, save this",
            "description": "",
            "user_message": "Yes, save this.",
        }],
        "operations": [],
    }))
    proposal = SimpleNamespace(content=json.dumps({
        "message": "This will update the thread preference.",
        "state": "proposal",
        "choices": [],
        "operations": [{
            "action": "create",
            "scope_type": "thread",
            "scope_id": "curator-thread",
            "content": "Answer in a funny way.",
            "override_targets": [],
        }],
    }))
    llm = SimpleNamespace(ainvoke=AsyncMock(side_effect=[permission, proposal]))
    monkeypatch.setattr(memory_curator_service, "get_llm", lambda *_args, **_kwargs: llm)

    response = await memory_curator_service.respond_to_memory_curator(
        MemoryCuratorRespondRequest(
            mode="create",
            context=_context(),
            messages=[MemoryCuratorMessage(role="user", content="Answer in a funny way.")],
            llm_model="chat-model",
            context_window=8192,
        )
    )

    assert response["state"] == "proposal"
    assert response["choices"] == []
    assert llm.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_memory_tools_enforce_capabilities_and_visible_scopes(curator_sessionmaker):
    await _workspace(curator_sessionmaker)
    created = await memory_curator_service.apply_memory_curator_change_set(
        MemoryCuratorApplyRequest(
            context=_context(),
            confirmed=True,
            operations=[MemoryCuratorOperation(
                action="create",
                scope_type="thread",
                scope_id="curator-thread",
                content="Use concise answers in this thread.",
            )],
        )
    )
    memory_id = created["changed_memories"][0]["id"]
    context, _thread, _project = await memory_tool_service.build_memory_tool_context(
        selected_scope_type="thread",
        selected_scope_id="curator-thread",
        thread_id="curator-thread",
        capabilities=[MEMORY_READ_STORED],
    )

    listed = await memory_tool_service.search_memory_tool(
        context,
        MemorySearchInput(view="stored", max_results=10),
    )
    fetched = await memory_tool_service.get_memory_tool(
        context,
        MemoryGetInput(memory_ids=[memory_id]),
    )

    assert [item["id"] for item in listed["memories"]] == [memory_id]
    assert fetched["memories"][0]["scope_type"] == "thread"
    context.capabilities = []
    with pytest.raises(memory_tool_service.MemoryToolPermissionError):
        await memory_tool_service.get_memory_tool(context, MemoryGetInput(memory_ids=[memory_id]))


@pytest.mark.asyncio
async def test_memory_tool_moves_global_memory_to_project_with_receipt(curator_sessionmaker):
    await _workspace(curator_sessionmaker)
    global_result = await memory_curator_service.apply_memory_curator_change_set(
        MemoryCuratorApplyRequest(
            context=_context(),
            confirmed=True,
            operations=[MemoryCuratorOperation(
                action="create",
                scope_type="user",
                scope_id="default",
                content="Research stocks and equities.",
            )],
        )
    )
    source = global_result["changed_memories"][0]
    context, _thread, _project = await memory_tool_service.build_memory_tool_context(
        selected_scope_type="project",
        selected_scope_id="curator-project",
        project_id="curator-project",
        capabilities=[MEMORY_PROPOSE],
    )
    prepared = await memory_tool_service.prepare_memory_change(
        context,
        MemoryPrepareChangeInput(intents=[{
            "action": "move",
            "memory_id": source["id"],
            "target_scope_type": "project",
            "override_target_ids": [],
        }]),
    )

    assert [item["action"] for item in prepared["operations"]] == ["create", "delete"]
    assert prepared["operation_summaries"][0]["action"] == "move"
    applied = await memory_tool_service.apply_confirmed_memory_change(
        MemoryCuratorApplyRequest(
            context=MemoryCuratorContext(
                selected_scope_type="project",
                selected_scope_id="curator-project",
                project_id="curator-project",
            ),
            confirmed=True,
            operations=[MemoryCuratorOperation.model_validate(item) for item in prepared["operations"]],
        )
    )

    assert applied["receipts"][0]["action"] == "move"
    assert applied["receipts"][0]["source_scope"]["scope_type"] == "user"
    assert applied["receipts"][0]["destination_scope"]["scope_type"] == "project"
    assert applied["receipts"][0]["deleted_memory_ids"] == [source["id"]]
    async with curator_sessionmaker() as session:
        assert await session.get(Memory, source["id"]) is None
        project_memories = list((await session.execute(
            select(Memory).where(Memory.scope_type == "project")
        )).scalars().all())
    assert len(project_memories) == 1
    assert project_memories[0].id == applied["receipts"][0]["result_memory_id"]


@pytest.mark.asyncio
async def test_memory_tool_move_reuses_identical_destination(curator_sessionmaker):
    await _workspace(curator_sessionmaker)
    created = []
    for scope_type, scope_id in (("user", "default"), ("project", "curator-project")):
        result = await memory_curator_service.apply_memory_curator_change_set(
            MemoryCuratorApplyRequest(
                context=_context(),
                confirmed=True,
                operations=[MemoryCuratorOperation(
                    action="create",
                    scope_type=scope_type,
                    scope_id=scope_id,
                    content="Prefer primary-source market research.",
                )],
            )
        )
        created.append(result["changed_memories"][0])
    source, destination = created
    context, _thread, _project = await memory_tool_service.build_memory_tool_context(
        selected_scope_type="project",
        selected_scope_id="curator-project",
        project_id="curator-project",
        capabilities=[MEMORY_PROPOSE],
    )
    prepared = await memory_tool_service.prepare_memory_change(
        context,
        MemoryPrepareChangeInput(intents=[{
            "action": "move",
            "memory_id": source["id"],
            "target_scope_type": "project",
            "override_target_ids": [],
        }]),
    )

    assert [item["action"] for item in prepared["operations"]] == ["update", "delete"]
    assert prepared["operation_summaries"][0]["destination_memory_id"] == destination["id"]


@pytest.mark.asyncio
async def test_curator_native_tool_call_prepares_proposal(curator_sessionmaker, monkeypatch):
    await _workspace(curator_sessionmaker)
    monkeypatch.setattr(memory_curator_service, "check_chat_model_ready", AsyncMock(return_value=True))
    monkeypatch.setattr(memory_curator_service, "check_model_supports_tools", AsyncMock(return_value=True))
    tool_call = AIMessage(content="", tool_calls=[{
        "name": "memory_prepare_change",
        "args": {"intents": [{
            "action": "create",
            "scope_type": "thread",
            "content": "Answer in a funny way.",
            "override_target_ids": [],
        }]},
        "id": "call-memory-1",
        "type": "tool_call",
    }])
    final = AIMessage(content=json.dumps({
        "message": "Create a thread preference for a funny tone.",
        "state": "proposal",
        "choices": [],
        "intents": [],
    }))
    bound = SimpleNamespace(ainvoke=AsyncMock(side_effect=[tool_call, final]))
    llm = SimpleNamespace(bind_tools=lambda _tools: bound, ainvoke=AsyncMock())
    monkeypatch.setattr(memory_curator_service, "get_llm", lambda *_args, **_kwargs: llm)

    response = await memory_curator_service.respond_to_memory_curator(
        MemoryCuratorRespondRequest(
            mode="create",
            context=_context(),
            messages=[MemoryCuratorMessage(role="user", content="Answer in a funny way.")],
            llm_model="tool-chat-model",
            context_window=8192,
        )
    )

    assert response["state"] == "proposal"
    assert response["tool_calls_used"] == 1
    assert response["operations"][0]["action"] == "create"
    assert response["operation_summaries"][0]["label"] == "Create thread memory"


@pytest.mark.asyncio
async def test_curator_ask_mode_requires_exact_query_approval(curator_sessionmaker, monkeypatch):
    await _workspace(curator_sessionmaker)
    monkeypatch.setattr(memory_curator_service, "check_chat_model_ready", AsyncMock(return_value=True))
    monkeypatch.setattr(memory_curator_service, "check_model_supports_tools", AsyncMock(return_value=True))
    tool_call = AIMessage(content="", tool_calls=[{
        "name": "internet_search",
        "args": {"query": "current Python packaging standard", "reason": "Verify its current name"},
        "id": "call-web-1",
        "type": "tool_call",
    }])
    final = AIMessage(content=json.dumps({
        "message": "Waiting for search approval.",
        "state": "clarification",
        "choices": [],
        "intents": [],
    }))
    bound = SimpleNamespace(ainvoke=AsyncMock(side_effect=[tool_call, final]))
    monkeypatch.setattr(
        memory_curator_service,
        "get_llm",
        lambda *_args, **_kwargs: SimpleNamespace(bind_tools=lambda _tools: bound, ainvoke=AsyncMock()),
    )
    search = AsyncMock()
    monkeypatch.setattr(memory_curator_service, "search_internet", search)

    response = await memory_curator_service.respond_to_memory_curator(
        MemoryCuratorRespondRequest(
            mode="create",
            context=_context(),
            messages=[MemoryCuratorMessage(role="user", content="Remember its current official name.")],
            llm_model="tool-chat-model",
            context_window=8192,
            web_search_mode="ask",
        )
    )

    assert response["state"] == "web_search_approval"
    assert response["pending_web_search"]["query"] == "current Python packaging standard"
    search.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirmed_web_provenance_is_persisted_without_snippet(curator_sessionmaker):
    await _workspace(curator_sessionmaker)
    result = await memory_curator_service.apply_memory_curator_change_set(
        MemoryCuratorApplyRequest(
            context=_context(),
            confirmed=True,
            operations=[MemoryCuratorOperation(
                action="create",
                scope_type="thread",
                scope_id="curator-thread",
                content="Use the current official packaging standard.",
                web_sources=[{
                    "id": "source-1",
                    "title": "Python Packaging User Guide",
                    "url": "https://packaging.python.org/",
                    "query": "current Python packaging standard",
                    "searched_at": "2026-08-03T12:00:00Z",
                }],
            )],
        )
    )
    async with curator_sessionmaker() as session:
        memory = await session.get(Memory, result["changed_memories"][0]["id"])
    assert memory.source_refs_json["web_sources"] == [{
        "id": "source-1",
        "title": "Python Packaging User Guide",
        "url": "https://packaging.python.org/",
        "query": "current Python packaging standard",
        "searched_at": "2026-08-03T12:00:00Z",
    }]


@pytest.mark.asyncio
async def test_curator_ignores_project_id_mistaken_for_override_memory_id(
    curator_sessionmaker,
    monkeypatch,
):
    await _workspace(curator_sessionmaker)
    monkeypatch.setattr(memory_curator_service, "check_chat_model_ready", AsyncMock(return_value=True))
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=json.dumps({
        "message": "Create this project memory.",
        "state": "proposal",
        "choices": [],
        "intents": [{
            "action": "create",
            "scope_type": "project",
            "content": "Research LLMs, AI, and machine learning for this project.",
            "override_target_ids": ["curator-project"],
        }],
    }))))
    monkeypatch.setattr(memory_curator_service, "get_llm", lambda *_args, **_kwargs: llm)

    response = await memory_curator_service.respond_to_memory_curator(
        MemoryCuratorRespondRequest(
            mode="create",
            context=MemoryCuratorContext(
                selected_scope_type="project",
                selected_scope_id="curator-project",
                project_id="curator-project",
            ),
            messages=[MemoryCuratorMessage(
                role="user",
                content="For this project, research LLMs, AI, and machine learning.",
            )],
            llm_model="chat-model",
            context_window=8192,
        )
    )

    assert response["state"] == "proposal"
    assert response["operations"][0]["scope_type"] == "project"
    assert response["operations"][0]["override_targets"] == []


@pytest.mark.asyncio
async def test_curator_asks_before_new_thread_memory_overrides_project_memory(
    curator_sessionmaker,
    monkeypatch,
):
    await _workspace(curator_sessionmaker)
    project_result = await memory_curator_service.apply_memory_curator_change_set(
        MemoryCuratorApplyRequest(
            context=_context(),
            confirmed=True,
            operations=[MemoryCuratorOperation(
                action="create",
                scope_type="project",
                scope_id="curator-project",
                content="Research AI, LLMs, and deep learning for this project.",
            )],
        )
    )
    project_memory_id = project_result["changed_memories"][0]["id"]
    monkeypatch.setattr(memory_curator_service, "check_chat_model_ready", AsyncMock(return_value=True))
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=json.dumps({
        "message": "Focus this thread on NVIDIA and its AI systems.",
        "state": "proposal",
        "choices": [],
        "intents": [{
            "action": "create",
            "scope_type": "thread",
            "content": "Focus on NVIDIA and its AI systems.",
            "override_target_ids": [project_memory_id],
        }],
    }))))
    monkeypatch.setattr(memory_curator_service, "get_llm", lambda *_args, **_kwargs: llm)

    response = await memory_curator_service.respond_to_memory_curator(
        MemoryCuratorRespondRequest(
            mode="create",
            context=_context(),
            messages=[MemoryCuratorMessage(
                role="user",
                content="For this thread, focus on NVIDIA and its AI systems.",
            )],
            llm_model="chat-model",
            context_window=8192,
        )
    )

    assert response["state"] == "conflict"
    assert response["operations"] == []
    assert [choice["id"] for choice in response["choices"]] == ["keep-both", "override-broader"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision_message", "decision_choice_id", "expected_override_count"),
    [
        (
            "Add this as an additional thread memory and keep the broader project memory effective. "
            "Do not override it.",
            "keep-both",
            0,
        ),
        ("Override the broader project memory in this thread context.", "override-broader", 1),
    ],
)
async def test_curator_honors_explicit_cross_scope_override_choice(
    curator_sessionmaker,
    monkeypatch,
    decision_message,
    decision_choice_id,
    expected_override_count,
):
    await _workspace(curator_sessionmaker)
    project_result = await memory_curator_service.apply_memory_curator_change_set(
        MemoryCuratorApplyRequest(
            context=_context(),
            confirmed=True,
            operations=[MemoryCuratorOperation(
                action="create",
                scope_type="project",
                scope_id="curator-project",
                content="Research AI, LLMs, and deep learning for this project.",
            )],
        )
    )
    project_memory_id = project_result["changed_memories"][0]["id"]
    monkeypatch.setattr(memory_curator_service, "check_chat_model_ready", AsyncMock(return_value=True))
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=json.dumps({
        "message": "Create the selected thread memory.",
        "state": "proposal",
        "choices": [],
        "intents": [{
            "action": "create",
            "scope_type": "thread",
            "content": "Focus on NVIDIA and its AI systems.",
            "override_target_ids": [project_memory_id],
        }],
    }))))
    monkeypatch.setattr(memory_curator_service, "get_llm", lambda *_args, **_kwargs: llm)

    response = await memory_curator_service.respond_to_memory_curator(
        MemoryCuratorRespondRequest(
            mode="create",
            context=_context(),
            messages=[MemoryCuratorMessage(
                role="user",
                content=decision_message,
                choice_id=decision_choice_id,
            )],
            llm_model="chat-model",
            context_window=8192,
        )
    )

    assert response["state"] == "proposal"
    assert len(response["operations"][0]["override_targets"]) == expected_override_count


@pytest.mark.asyncio
async def test_curator_additive_choice_removes_existing_cross_scope_override(
    curator_sessionmaker,
    monkeypatch,
):
    await _workspace(curator_sessionmaker)
    project_result = await memory_curator_service.apply_memory_curator_change_set(
        MemoryCuratorApplyRequest(
            context=_context(),
            confirmed=True,
            operations=[MemoryCuratorOperation(
                action="create",
                scope_type="project",
                scope_id="curator-project",
                content="Research AI, LLMs, and deep learning for this project.",
            )],
        )
    )
    project_memory = project_result["changed_memories"][0]
    thread_result = await memory_curator_service.apply_memory_curator_change_set(
        MemoryCuratorApplyRequest(
            context=_context(),
            confirmed=True,
            operations=[MemoryCuratorOperation(
                action="create",
                scope_type="thread",
                scope_id="curator-thread",
                content="Focus on NVIDIA and its AI systems.",
                override_targets=[{
                    "memory_id": project_memory["id"],
                    "expected_updated_at": project_memory["updated_at"],
                }],
            )],
        )
    )
    thread_memory = thread_result["changed_memories"][0]
    monkeypatch.setattr(memory_curator_service, "check_chat_model_ready", AsyncMock(return_value=True))
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=json.dumps({
        "message": "Keep both memories.",
        "state": "proposal",
        "choices": [],
        "intents": [{
            "action": "set_overrides",
            "memory_id": thread_memory["id"],
            "override_target_ids": [project_memory["id"]],
        }],
    }))))
    monkeypatch.setattr(memory_curator_service, "get_llm", lambda *_args, **_kwargs: llm)

    response = await memory_curator_service.respond_to_memory_curator(
        MemoryCuratorRespondRequest(
            mode="edit",
            context=_context(),
            memory_id=thread_memory["id"],
            messages=[MemoryCuratorMessage(
                role="user",
                content="Keep both memories effective. Do not override the broader project memory.",
                choice_id="keep-both",
            )],
            llm_model="chat-model",
            context_window=8192,
        )
    )

    assert response["state"] == "proposal"
    assert response["operations"][0]["override_targets"] == []
