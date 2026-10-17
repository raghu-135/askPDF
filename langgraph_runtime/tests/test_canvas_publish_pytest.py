import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import ValidationError

from langgraph_runtime.mcp_client import _PublishCanvasArguments, create_mcp_langchain_tool
from langgraph_runtime.workflows.canvas_publish import (
    CanvasSpecNormalizeError,
    canvas_idempotency_key,
    prepare_publish_canvas_tool_input,
    sanitize_supersedes_id_for_publish,
    strip_false_canvas_absence_prose,
    strip_prose_publish_canvas,
    strip_unpublished_canvas_claims,
    synthesize_with_canvas_publish,
)


def test_publish_canvas_schema_requires_draft_fields():
    schema = create_mcp_langchain_tool("publish_canvas").args_schema.model_json_schema()
    required = set(schema.get("required", []))
    assert {"layout", "title", "claims"}.issubset(required)
    assert "spec" not in required
    assert "query" not in schema.get("properties", {})


@pytest.mark.asyncio
async def test_publish_canvas_rejects_query_only_arguments():
    tool = create_mcp_langchain_tool("publish_canvas")
    with pytest.raises(ValidationError):
        await tool.ainvoke({"query": "make a canvas"})


def test_canvas_idempotency_key_distinguishes_create_and_revision():
    state = {"agent_run_id": "run-42"}
    assert canvas_idempotency_key(state) == "publish_canvas:run-42:create"
    assert canvas_idempotency_key(state, supersedes_id="canvas-1") == "publish_canvas:run-42:canvas-1"


def test_prepare_publish_canvas_tool_input_requires_claims():
    with pytest.raises(CanvasSpecNormalizeError):
        prepare_publish_canvas_tool_input({"layout": "matrix", "title": "X"})


def test_prepare_publish_canvas_tool_input_jsonifies_pydantic_claims():
    class Claim:
        def model_dump(self, mode="json", exclude_none=True):
            return {"text": "Skill", "source_ids": ["web:https://example.com"]}

    payload = prepare_publish_canvas_tool_input({
        "layout": "matrix",
        "title": "Skills",
        "claims": [Claim()],
    })
    assert payload["claims"] == [{"text": "Skill", "source_ids": ["web:https://example.com"]}]


def test_prepare_publish_canvas_tool_input_accepts_draft():
    payload = prepare_publish_canvas_tool_input({
        "layout": "timeline",
        "title": "Events",
        "claims": [{"text": "Happened", "source_ids": ["web:https://example.com"], "time": "2020"}],
    })
    assert payload["layout"] == "timeline"
    assert len(payload["claims"]) == 1


def test_publish_canvas_arguments_accept_draft():
    args = _PublishCanvasArguments(
        layout="matrix",
        title="Resume Skills Matrix",
        claims=[{"text": "Skill A", "source_ids": ["web:https://example.com/a"]}],
    )
    assert args.title == "Resume Skills Matrix"
    assert args.claims[0].text == "Skill A"


def test_strip_prose_publish_canvas_removes_fake_calls():
    text = 'Findings.\n\npublish_canvas(title="Research Summary")\n\nDone.'
    assert "publish_canvas" not in strip_prose_publish_canvas(text)
    assert "Findings." in strip_prose_publish_canvas(text)


def test_strip_false_canvas_absence_prose_removes_denials_when_draft_exists():
    text = (
        "Summary of findings.\n\n"
        "No research canvas content provided.\n"
        "No document citations.\n"
        "Transformer BLEU is 28.4."
    )
    cleaned = strip_false_canvas_absence_prose(text)
    assert "no research canvas" not in cleaned.casefold()
    assert "no document citations" not in cleaned.casefold()
    assert "28.4" in cleaned


def test_strip_unpublished_canvas_claims_removes_false_publish_narration():
    text = "Here is a research canvas comparing the two papers.\n\nThe BLEU score is 28.4."
    cleaned = strip_unpublished_canvas_claims(text)
    assert "research canvas" not in cleaned.casefold()
    assert "28.4" in cleaned



@pytest.mark.asyncio
async def test_helper_skips_mcp_when_spec_normalizes_to_empty_sections(monkeypatch):
    tool_calls = []

    async def fake_llm(func, messages, **_kwargs):
        return await func(messages)

    async def fake_tool(name, tool_input, **_kwargs):
        tool_calls.append((name, tool_input))
        return {"ok": True, "content": "ok", "error": None, "trace": {"tool_name": name}}

    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.invoke_llm_for_node", fake_llm)
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.invoke_tool_for_node", fake_tool)
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.append_tool_event_for_node", lambda *args, **kwargs: [])
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.tool_config_for_node", lambda *args, **kwargs: {"configurable": {"tool_call_id": "call-1"}})
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.normalize_tool_result", lambda raw, **kwargs: raw)

    class Bound:
        async def ainvoke(self, messages):
            if any(isinstance(item, ToolMessage) for item in messages):
                return SimpleNamespace(content="Prose only.", tool_calls=[], usage_metadata={})
            return SimpleNamespace(
                content="",
                tool_calls=[{
                    "name": "publish_canvas",
                    "id": "call-1",
                    "args": {"layout": "matrix", "title": "X"},
                }],
                usage_metadata={},
            )

    class LLM:
        def bind_tools(self, _tools):
            return Bound()

        async def ainvoke(self, _messages):
            raise AssertionError("unbound model must not run while emit is admitted")

    result = await synthesize_with_canvas_publish(
        LLM(),
        [HumanMessage(content="Compare")],
        state={"allowed_tool_ids": ["research_canvas_publish"], "agent_run_id": "run-1", "llm_model": "test"},
        config={},
        node="synthesizer",
        started=0.0,
    )
    assert result["published"] is False
    assert tool_calls == []
    assert "could not be published" in result["answer"]


@pytest.mark.asyncio
async def test_helper_does_not_invoke_publish_when_not_admitted(monkeypatch):
    async def fake_llm(func, messages, **_kwargs):
        return await func(messages)

    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.invoke_llm_for_node", fake_llm)
    monkeypatch.setattr(
        "langgraph_runtime.workflows.canvas_publish.invoke_tool_for_node",
        AsyncMock(side_effect=AssertionError("publish_canvas must not run")),
    )

    class LLM:
        def bind_tools(self, _tools):
            raise AssertionError("tools must not bind when canvas emit is not admitted")

        async def ainvoke(self, _messages):
            return SimpleNamespace(content="Prose answer", tool_calls=[], usage_metadata={})

    result = await synthesize_with_canvas_publish(
        LLM(),
        [HumanMessage(content="Summarize")],
        state={"allowed_tool_ids": ["document_search_knowledge"], "llm_model": "test"},
        config={},
        node="synthesizer",
        started=0.0,
    )
    assert result["published"] is False
    assert result["answer"] == "Prose answer"


@pytest.mark.asyncio
async def test_helper_invokes_publish_canvas_once_with_draft(monkeypatch):
    tool_calls = []

    async def fake_llm(func, messages, **_kwargs):
        return await func(messages)

    async def fake_tool(name, tool_input, **_kwargs):
        tool_calls.append((name, tool_input))
        return {"ok": True, "content": "published Advisor Models", "error": None, "trace": {"tool_name": name}}

    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.invoke_llm_for_node", fake_llm)
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.invoke_tool_for_node", fake_tool)
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.append_tool_event_for_node", lambda *args, **kwargs: [])
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.tool_config_for_node", lambda *args, **kwargs: {})
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.normalize_tool_result", lambda raw, **kwargs: raw)

    draft = {
        "layout": "matrix",
        "title": "Advisor Models",
        "claims": [{"text": "Claim one", "source_ids": ["web:https://example.com/paper"]}],
    }

    class Bound:
        async def ainvoke(self, messages):
            if any(isinstance(item, ToolMessage) for item in messages):
                raise AssertionError("bound model should unbind after a successful publish")
            return SimpleNamespace(
                content="",
                tool_calls=[{"name": "publish_canvas", "id": "call-1", "args": draft}],
                usage_metadata={},
            )

    class LLM:
        def bind_tools(self, _tools):
            return Bound()

        async def ainvoke(self, _messages):
            return SimpleNamespace(content="See the canvas Advisor Models.", tool_calls=[], usage_metadata={})

    result = await synthesize_with_canvas_publish(
        LLM(),
        [HumanMessage(content="Compare")],
        state={"allowed_tool_ids": ["research_canvas_publish"], "agent_run_id": "run-1", "llm_model": "test"},
        config={},
        node="synthesizer",
        started=0.0,
    )
    assert result["published"] is True
    assert result["answer"] == "See the canvas Advisor Models."
    assert tool_calls == [
        (
            "publish_canvas",
            {
                **draft,
                "idempotency_key": "publish_canvas:run-1:create",
            },
        )
    ]


@pytest.mark.asyncio
async def test_helper_rejects_legacy_block_shaped_args(monkeypatch):
    tool_calls = []

    async def fake_llm(func, messages, **_kwargs):
        return await func(messages)

    async def fake_tool(name, tool_input, **_kwargs):
        tool_calls.append((name, tool_input))
        return {"ok": True, "content": "published", "error": None, "trace": {"tool_name": name}}

    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.invoke_llm_for_node", fake_llm)
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.invoke_tool_for_node", fake_tool)
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.append_tool_event_for_node", lambda *args, **kwargs: [])
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.tool_config_for_node", lambda *args, **kwargs: {})
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.normalize_tool_result", lambda raw, **kwargs: raw)

    class Bound:
        async def ainvoke(self, messages):
            if any(isinstance(item, ToolMessage) for item in messages):
                return SimpleNamespace(content="See the canvas.", tool_calls=[], usage_metadata={})
            return SimpleNamespace(
                content="",
                tool_calls=[{"name": "publish_canvas", "id": "call-1", "args": {"type": "markdown", "content": "A claim"}}],
                usage_metadata={},
            )

    class LLM:
        def bind_tools(self, _tools):
            return Bound()

        async def ainvoke(self, _messages):
            return SimpleNamespace(content="See the canvas.", tool_calls=[], usage_metadata={})

    result = await synthesize_with_canvas_publish(
        LLM(),
        [HumanMessage(content="Compare")],
        state={"allowed_tool_ids": ["research_canvas_publish"], "agent_run_id": "run-1", "llm_model": "test"},
        config={},
        node="deep_task_synthesizer",
        started=0.0,
    )
    assert result["published"] is False
    assert tool_calls == []


@pytest.mark.asyncio
async def test_helper_strips_fake_call_after_admission_failure(monkeypatch):
    async def fake_llm(func, messages, **_kwargs):
        return await func(messages)

    async def fake_tool(_name, _tool_input, **_kwargs):
        return {"ok": False, "content": "missing citations", "error": {"code": "publish_canvas_missing_citations"}, "trace": {}}

    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.invoke_llm_for_node", fake_llm)
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.invoke_tool_for_node", fake_tool)
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.append_tool_event_for_node", lambda *args, **kwargs: [])
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.tool_config_for_node", lambda *args, **kwargs: {})
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.normalize_tool_result", lambda raw, **kwargs: raw)

    class Bound:
        async def ainvoke(self, messages):
            if any(isinstance(item, ToolMessage) for item in messages):
                return SimpleNamespace(
                    content='Fallback.\n\npublish_canvas(title="X")\n',
                    tool_calls=[],
                    usage_metadata={},
                )
            return SimpleNamespace(
                content="",
                tool_calls=[{
                    "name": "publish_canvas",
                    "id": "call-1",
                    "args": {
                        "layout": "matrix",
                        "title": "X",
                        "claims": [{"text": "Claim", "source_ids": ["web:https://example.com/x"]}],
                    },
                }],
                usage_metadata={},
            )

    class LLM:
        def bind_tools(self, _tools):
            return Bound()

        async def ainvoke(self, _messages):
            raise AssertionError("unbound model must not run while emit is admitted")

    result = await synthesize_with_canvas_publish(
        LLM(),
        [HumanMessage(content="Compare")],
        state={"allowed_tool_ids": ["research_canvas_publish"], "agent_run_id": "run-1", "llm_model": "test"},
        config={},
        node="deep_task_synthesizer",
        started=0.0,
    )
    assert result["published"] is False
    assert "publish_canvas" not in result["answer"]
    assert "Fallback." in result["answer"]
    assert "could not be published" in result["answer"]


@pytest.mark.asyncio
async def test_failed_publish_with_empty_content_reinvokes_unbound_for_prose(monkeypatch):
    unbound_calls = 0

    async def fake_llm(func, messages, **_kwargs):
        return await func(messages)

    async def fake_tool(_name, _tool_input, **_kwargs):
        return {"ok": False, "content": "invalid spec", "error": {"code": "publish_canvas_invalid_spec"}, "trace": {}}

    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.invoke_llm_for_node", fake_llm)
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.invoke_tool_for_node", fake_tool)
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.append_tool_event_for_node", lambda *args, **kwargs: [])
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.tool_config_for_node", lambda *args, **kwargs: {})
    monkeypatch.setattr("langgraph_runtime.workflows.canvas_publish.normalize_tool_result", lambda raw, **kwargs: raw)

    class Bound:
        async def ainvoke(self, _messages):
            return SimpleNamespace(
                content="",
                tool_calls=[{
                    "name": "publish_canvas",
                    "id": "call-1",
                    "args": {
                        "layout": "matrix",
                        "title": "X",
                        "claims": [{"text": "Claim", "source_ids": ["web:https://example.com/x"]}],
                    },
                }],
                usage_metadata={},
            )

    class LLM:
        def bind_tools(self, _tools):
            return Bound()

        async def ainvoke(self, _messages):
            nonlocal unbound_calls
            unbound_calls += 1
            return SimpleNamespace(content="Factual answer from recovery.", tool_calls=[], usage_metadata={})

    result = await synthesize_with_canvas_publish(
        LLM(),
        [HumanMessage(content="What year was BERT published?")],
        state={"allowed_tool_ids": ["research_canvas_publish"], "agent_run_id": "run-1", "llm_model": "test"},
        config={},
        node="synthesizer",
        started=0.0,
    )
    assert result["published"] is False
    assert unbound_calls == 1
    assert "Factual answer from recovery." in result["answer"]
    assert "could not be published" in result["answer"]


def test_sanitize_supersedes_id_drops_file_hash_and_garbage():
    file_hash = "3deaaae75be4cddfd0b34d7568ae5635"
    assert sanitize_supersedes_id_for_publish(file_hash) is None
    assert sanitize_supersedes_id_for_publish("Resume Skills Matrix") is None
    canvas_id = "41a7cdc2-1234-5678-9abc-def012345678"
    assert sanitize_supersedes_id_for_publish(canvas_id) == canvas_id

