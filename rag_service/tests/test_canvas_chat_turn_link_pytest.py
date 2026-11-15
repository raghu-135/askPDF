import uuid
from unittest.mock import AsyncMock

import pytest
from types import SimpleNamespace

from app.services.canvas_service import CanvasService


@pytest.mark.asyncio
async def test_link_published_canvases_to_turn_assigns_agent_run(monkeypatch):
    assign = AsyncMock(return_value=1)
    canvases = SimpleNamespace(assign_chat_turn_for_agent_run=assign)
    service = CanvasService(canvases=canvases, threads=SimpleNamespace(), messages=SimpleNamespace())

    linked = await service.link_published_canvases_to_turn(
        "thread-1",
        agent_run_id="run-a",
        chat_turn_id="turn-a",
    )

    assert linked == 1
    assign.assert_awaited_once_with("thread-1", agent_run_id="run-a", chat_turn_id="turn-a")


@pytest.mark.asyncio
async def test_refs_by_turn_maps_each_assistant_turn_to_its_canvas(monkeypatch):
    run_create = "run-create"
    run_update = "run-update"
    turn_create = "turn-create"
    turn_update = "turn-update"
    canvas_v1 = str(uuid.uuid4())
    canvas_v2 = str(uuid.uuid4())

    class FakeCanvases:
        async def list_for_thread(self, thread_id: str, *, current_only: bool = True):
            return [
                SimpleNamespace(
                    id=canvas_v2,
                    agent_run_id=run_update,
                    chat_turn_id=turn_update,
                    created_at=2,
                    summary_json={"title": "V2"},
                    object_key="k2",
                ),
                SimpleNamespace(
                    id=canvas_v1,
                    agent_run_id=run_create,
                    chat_turn_id=None,
                    created_at=1,
                    summary_json={"title": "V1"},
                    object_key="k1",
                ),
            ]

    class FakeMessages:
        async def get_turn_by_agent_run_id(self, thread_id: str, agent_run_id: str):
            if agent_run_id == run_create:
                return SimpleNamespace(id=turn_create)
            if agent_run_id == run_update:
                return SimpleNamespace(id=turn_update)
            return None

    service = CanvasService(canvases=FakeCanvases(), threads=SimpleNamespace(), messages=FakeMessages())
    refs = await service.refs_by_turn("thread-1")
    assert refs[turn_create]["id"] == canvas_v1
    assert refs[turn_update]["id"] == canvas_v2