from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.models.canvas import admit_canvas_spec, parse_canvas_spec
from app.services.tool_approval import invocation_policies
from app.tools.context import ToolInvocationContext
from app.tools.publish_canvas import PublishCanvasRequest, invoke_publish_canvas
from tests.test_canvas_spec_pytest import valid_spec


def test_admit_canvas_spec_rejects_missing_citations():
    payload = {
        "schema_version": 1,
        "title": "No citations",
        "sections": [
            {
                "title": "Empty",
                "blocks": [{"type": "stat", "value": "1", "label": "One"}],
            }
        ],
    }
    spec = parse_canvas_spec(payload)
    try:
        admit_canvas_spec(spec, thread_file_hashes={"file-a"})
    except Exception as exc:
        assert "missing_citations" in getattr(exc, "code", "") or "citation" in str(exc)
    else:
        raise AssertionError("canvases without citations must be rejected")


def test_admit_canvas_spec_rejects_unknown_document_hash():
    spec = parse_canvas_spec(valid_spec())
    try:
        admit_canvas_spec(spec, thread_file_hashes={"other-file"})
    except Exception as exc:
        assert "unknown_document_citation" in getattr(exc, "code", "") or "file-a" in str(exc)
    else:
        raise AssertionError("citations must resolve to thread documents")


def test_hitl_canvas_publish_asks_before_durable_write():
    policies = invocation_policies({"hitl_canvas_publish": True})
    assert policies["publish_canvas"] == {"mode": "ask", "scope": "run"}
    assert "publish_canvas" not in invocation_policies({})


@pytest.mark.asyncio
async def test_publish_canvas_rejects_invalid_components(monkeypatch):
    monkeypatch.setattr("app.tools.publish_canvas.get_thread_files", AsyncMock(return_value=[]))
    with pytest.raises(ValidationError):
        PublishCanvasRequest.model_validate(
            {"spec": {**valid_spec(), "sections": [{"title": "Bad", "blocks": [{"type": "iframe", "src": "https://evil.example"}]}]}}
        )
    result = await invoke_publish_canvas(
        PublishCanvasRequest.model_validate({"spec": valid_spec()}),
        ToolInvocationContext(thread_id="thread-1", run_id="run-1"),
    )
    assert result.ok is False
    assert result.error.code == "publish_canvas_unknown_document_citation"


@pytest.mark.asyncio
async def test_publish_canvas_rejects_missing_citations(monkeypatch):
    monkeypatch.setattr("app.tools.publish_canvas.get_thread_files", AsyncMock(return_value=[SimpleNamespace(file_hash="file-a")]))
    payload = {
        "schema_version": 1,
        "title": "No citations",
        "sections": [
            {
                "title": "Empty",
                "blocks": [{"type": "markdown", "text": "Summary only."}],
            }
        ],
    }
    result = await invoke_publish_canvas(
        PublishCanvasRequest.model_validate({"spec": payload}),
        ToolInvocationContext(thread_id="thread-1", run_id="run-1"),
    )
    assert result.ok is False
    assert result.error.code == "publish_canvas_missing_citations"


@pytest.mark.asyncio
async def test_publish_canvas_ignores_file_hash_supersedes_id(monkeypatch):
    monkeypatch.setattr(
        "app.tools.publish_canvas.get_thread_files",
        AsyncMock(return_value=[SimpleNamespace(file_hash="file-a")]),
    )
    create = AsyncMock(return_value={"id": "canvas-1", "title": "Skills"})
    monkeypatch.setattr("app.tools.publish_canvas.CanvasService.create", create)
    file_hash = "3deaaae75be4cddfd0b34d7568ae5635"
    result = await invoke_publish_canvas(
        PublishCanvasRequest.model_validate({"spec": valid_spec(), "supersedes_id": file_hash}),
        ToolInvocationContext(thread_id="thread-1", run_id="run-1"),
    )
    assert result.ok is True
    assert "supersedes_id_ignored_file_hash" in (result.warnings or [])
    create.assert_awaited_once()
    assert create.await_args[0][1].supersedes_id is None


@pytest.mark.asyncio
async def test_publish_canvas_persists_admitted_spec(monkeypatch):
    monkeypatch.setattr(
        "app.tools.publish_canvas.get_thread_files",
        AsyncMock(
            return_value=[
                SimpleNamespace(file_hash="file-a"),
                SimpleNamespace(file_hash="file-b"),
            ]
        ),
    )
    created = {"id": "canvas-1", "title": "Compare the two papers"}
    monkeypatch.setattr(
        "app.tools.publish_canvas.CanvasService.create",
        AsyncMock(return_value=created),
    )
    result = await invoke_publish_canvas(
        PublishCanvasRequest.model_validate({"spec": valid_spec()}),
        ToolInvocationContext(thread_id="thread-1", run_id="run-1"),
    )
    assert result.ok is True
    assert result.artifacts["canvas"]["id"] == "canvas-1"
    assert "Published research canvas" in result.content
