import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from app.models.memory_limits import MAX_MEMORY_QUERY_CHARS, MAX_MEMORY_ROWS
from app.models.memory_tools import (
    MemoryAttributes,
    MemorySearchInput,
    normalize_memory_attributes,
)
from app.models.requests import MemorySearchRequest
from app.services import (
    memory_repair_scheduler,
    memory_representation_service,
    memory_service,
    memory_workspace_service,
    web_search_service,
)


class _EmptySession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def begin(self):
        return self

    async def execute(self, _statement):
        return SimpleNamespace(
            scalars=lambda: SimpleNamespace(all=lambda: []),
        )


@pytest.mark.asyncio
async def test_web_search_normalizes_ddgs_results_and_bounds_provider(monkeypatch):
    calls = []

    def provider(query, max_results):
        calls.append((query, max_results))
        return [
            {"title": "First", "href": "https://example.test/one", "body": "First body"},
            {"title": "No body", "href": "https://example.test/empty"},
            {"title": "Second", "href": "https://example.test/two", "body": "Second body"},
        ]

    async def rerank(_query, chunks):
        return [{**chunk, "rerank_score": 0.9 - index / 10} for index, chunk in enumerate(chunks)]

    monkeypatch.setattr(web_search_service, "_search_provider", provider)
    monkeypatch.setattr(web_search_service, "rerank_document_chunks", rerank)
    monkeypatch.setattr(web_search_service, "iso_utc_z", lambda *_args: "2026-08-05T00:00:00Z")

    result = await web_search_service.search_internet("  current topic  ", max_results=2)

    assert calls == [("current topic", 2)]
    assert result["query"] == "current topic"
    assert [source["title"] for source in result["sources"]] == ["First", "Second"]
    assert result["sources"][0]["url"] == "https://example.test/one"
    assert result["sources"][0]["score"] == 0.9
    assert len(result["sources"][0]["id"]) == 24

    repeated = await web_search_service.search_internet("current topic", max_results=2)
    assert repeated["sources"][0]["id"] == result["sources"][0]["id"]


@pytest.mark.asyncio
async def test_web_search_empty_results_and_provider_errors_are_explicit(monkeypatch):
    monkeypatch.setattr(web_search_service, "_search_provider", lambda *_args: [])
    empty = await web_search_service.search_internet("nothing", use_reranker=False)
    assert empty["sources"] == []

    def fail(*_args):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(web_search_service, "_search_provider", fail)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await web_search_service.search_internet("failure", use_reranker=False)

    with pytest.raises(ValueError, match="max_results must be positive"):
        await web_search_service.search_internet("bounded", max_results=0)


@pytest.mark.asyncio
async def test_memory_repair_scheduler_deduplicates_by_key(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def repair(_model):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {"indexed_ids": []}

    monkeypatch.setattr(memory_representation_service, "warm_global_representations_for_model", repair)
    first = memory_repair_scheduler.schedule_global_representation_repair("model-a")
    await started.wait()
    second = memory_repair_scheduler.schedule_global_representation_repair("model-a")

    assert first is second
    assert calls == 1
    assert memory_repair_scheduler.pending_memory_repair_keys() == ("global-representations:model-a",)

    release.set()
    await first
    await asyncio.sleep(0)
    assert memory_repair_scheduler.pending_memory_repair_keys() == ()


@pytest.mark.asyncio
async def test_memory_repair_shutdown_cancels_owned_tasks():
    started = asyncio.Event()

    async def repair():
        started.set()
        await asyncio.Event().wait()

    task = memory_repair_scheduler.schedule_memory_repair("shutdown-test", repair)
    await started.wait()
    await memory_repair_scheduler.shutdown_memory_repairs()

    assert task.cancelled()
    assert memory_repair_scheduler.pending_memory_repair_keys() == ()


@pytest.mark.asyncio
async def test_failed_memory_repair_is_consumed_and_can_be_retried():
    attempts = 0

    async def repair():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("repair failed")
        return "repaired"

    first = memory_repair_scheduler.schedule_memory_repair("retry-test", repair)
    assert isinstance((await asyncio.gather(first, return_exceptions=True))[0], RuntimeError)
    await asyncio.sleep(0)
    assert memory_repair_scheduler.pending_memory_repair_keys() == ()

    second = memory_repair_scheduler.schedule_memory_repair("retry-test", repair)
    assert await second == "repaired"
    assert attempts == 2


def test_memory_attributes_are_canonical_for_partial_and_invalid_input():
    assert normalize_memory_attributes({"kind": "preference"}) == {
        "kind": "preference",
        "applicability": ["task_specific"],
        "durability": "stable",
    }
    assert normalize_memory_attributes({"kind": "unknown"}) == MemoryAttributes().model_dump(mode="json")


def test_shared_memory_validation_limits_accept_boundary_and_reject_overflow():
    MemorySearchInput(query="x" * MAX_MEMORY_QUERY_CHARS, max_results=MAX_MEMORY_ROWS)
    MemorySearchRequest(query="x" * MAX_MEMORY_QUERY_CHARS, max_results=50)

    with pytest.raises(ValidationError):
        MemorySearchInput(query="x" * (MAX_MEMORY_QUERY_CHARS + 1))
    with pytest.raises(ValidationError):
        MemorySearchInput(max_results=MAX_MEMORY_ROWS + 1)
    with pytest.raises(ValidationError):
        MemorySearchRequest(query="x" * (MAX_MEMORY_QUERY_CHARS + 1))


def test_application_code_uses_pydantic_v2_serialization():
    app_dir = Path(__file__).parents[1] / "app"
    offenders = []
    for path in app_dir.rglob("*.py"):
        if ".dict(" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(app_dir)))
    assert offenders == []


@pytest.mark.asyncio
async def test_thread_memory_cleanup_uses_scope_wide_delete_without_row_truncation(monkeypatch):
    list_memories = AsyncMock(return_value=[SimpleNamespace(id="first")])
    delete_scope = AsyncMock(return_value=[f"memory-{index}" for index in range(600)])
    delete_vectors = AsyncMock(return_value=True)

    monkeypatch.setattr(memory_service, "list_memories", list_memories)
    monkeypatch.setattr(memory_service, "delete_memories_for_scope", delete_scope)
    monkeypatch.setattr(
        memory_service,
        "resolve_thread_embedding_context",
        AsyncMock(return_value=SimpleNamespace(embedding_model="model-a")),
    )
    monkeypatch.setattr(
        memory_service,
        "get_vector_db",
        lambda: SimpleNamespace(delete_memory_vectors_for_scope=delete_vectors),
    )

    result = await memory_service.hard_delete_thread_memory_resources("thread-a")

    list_memories.assert_awaited_once_with(scope_type="thread", scope_id="thread-a", limit=1)
    delete_scope.assert_awaited_once_with(scope_type="thread", scope_id="thread-a")
    assert len(result["deleted_memory_ids"]) == 600


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_ready", "representation_ready", "expected_status"),
    [(False, True, "blocked"), (True, True, "ready"), (True, False, "indexing")],
)
async def test_memory_workspace_readiness_contract(
    monkeypatch,
    model_ready,
    representation_ready,
    expected_status,
):
    representation_status = {
        "ready": representation_ready,
        "total_count": 0 if representation_ready else 1,
        "indexed_count": 0,
        "pending_count": 0 if representation_ready else 1,
        "failed_count": 0,
    }
    monkeypatch.setattr(
        memory_workspace_service,
        "_resolve_context",
        AsyncMock(return_value=("global", "model-a", [("user", "default")])),
    )
    monkeypatch.setattr(
        memory_workspace_service,
        "check_embedding_model_ready",
        AsyncMock(return_value=model_ready),
    )
    monkeypatch.setattr(
        memory_workspace_service,
        "global_representation_status_for_model",
        AsyncMock(return_value=representation_status),
    )
    monkeypatch.setattr(memory_workspace_service, "async_session_maker", lambda: _EmptySession())

    result = await memory_workspace_service.get_memory_workspace_readiness()

    assert result["status"] == expected_status
    assert result["ready"] is (expected_status == "ready")
    assert result["canonical"] == {
        "total_count": 0,
        "indexed": 0,
        "pending": 0,
        "indexing": 0,
        "failed": 0,
    }
    assert result["global_representations"] == representation_status


@pytest.mark.asyncio
async def test_memory_workspace_prepare_schedules_representation_repair(monkeypatch):
    schedule = Mock()
    monkeypatch.setattr(
        memory_workspace_service,
        "_resolve_context",
        AsyncMock(return_value=("project", "model-a", [("project", "project-a")])),
    )
    monkeypatch.setattr(
        memory_workspace_service,
        "check_embedding_model_ready",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        memory_workspace_service,
        "reset_stale_global_representation_indexes",
        AsyncMock(return_value=0),
    )
    monkeypatch.setattr(
        memory_workspace_service,
        "global_representation_status_for_model",
        AsyncMock(return_value={
            "ready": False,
            "total_count": 1,
            "indexed_count": 0,
            "pending_count": 1,
            "failed_count": 0,
        }),
    )
    monkeypatch.setattr(memory_workspace_service, "schedule_global_representation_repair", schedule)
    monkeypatch.setattr(memory_workspace_service, "async_session_maker", lambda: _EmptySession())

    result = await memory_workspace_service.get_memory_workspace_readiness(
        project_id="project-a",
        prepare=True,
    )

    assert result["status"] == "indexing"
    schedule.assert_called_once_with("model-a")
