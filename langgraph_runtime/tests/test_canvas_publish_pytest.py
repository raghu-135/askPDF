from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import ValidationError

from langgraph_runtime.mcp_client import create_mcp_langchain_tool
from langgraph_runtime.workflows.canvas_publish import (
    normalize_publish_canvas_spec,
    strip_prose_publish_canvas,
    synthesize_with_canvas_publish,
)


def test_publish_canvas_schema_requires_spec():
    schema = create_mcp_langchain_tool("publish_canvas").args_schema.model_json_schema()
    assert "spec" in schema.get("required", [])
    assert "query" not in schema.get("properties", {})


@pytest.mark.asyncio
async def test_publish_canvas_rejects_query_only_arguments():
    tool = create_mcp_langchain_tool("publish_canvas")
    with pytest.raises(ValidationError):
        await tool.ainvoke({"query": "make a canvas"})


def test_strip_prose_publish_canvas_removes_fake_calls():
    text = 'Findings.\n\npublish_canvas(title="Research Summary")\n\nDone.'
    assert "publish_canvas" not in strip_prose_publish_canvas(text)
    assert "Findings." in strip_prose_publish_canvas(text)


def test_normalize_publish_canvas_spec_wraps_a_block_as_canvas_spec_v1():
    spec = normalize_publish_canvas_spec({"type": "markdown", "content": "Advisor models steer black-box LLMs."})
    assert spec["schema_version"] == 1
    assert spec["sections"][0]["blocks"][0] == {"type": "markdown", "text": "Advisor models steer black-box LLMs."}


def test_normalize_publish_canvas_spec_splits_markdown_and_sources_extras():
    spec = normalize_publish_canvas_spec({
        "type": "markdown",
        "content": "Advisor models steer black-box LLMs.",
        "sources": [{"kind": "web", "label": "Paper", "url": "https://example.com"}],
        "items": [{"kind": "web", "label": "ignored"}],
    })
    blocks = spec["sections"][0]["blocks"]
    assert blocks[0] == {"type": "markdown", "text": "Advisor models steer black-box LLMs."}
    assert blocks[1] == {
        "type": "sources",
        "citations": [{"kind": "web", "label": "Paper", "url": "https://example.com"}],
    }


def test_normalize_publish_canvas_spec_preserves_canvas_spec_v1():
    original = {"schema_version": 1, "title": "Advisor Models", "sections": [{"title": "Coverage", "blocks": [{"type": "stat", "value": "3", "label": "Claims"}]}]}
    assert normalize_publish_canvas_spec({"spec": original}) == original


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
async def test_helper_invokes_publish_canvas_once_with_spec(monkeypatch):
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

    spec = {"schema_version": 1, "title": "Advisor Models", "sections": []}

    class Bound:
        async def ainvoke(self, messages):
            if any(isinstance(item, ToolMessage) for item in messages):
                raise AssertionError("bound model should unbind after a successful publish")
            return SimpleNamespace(
                content="",
                tool_calls=[{"name": "publish_canvas", "id": "call-1", "args": {"spec": spec}}],
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
    assert tool_calls == [("publish_canvas", {"spec": spec, "idempotency_key": "run-1"})]


@pytest.mark.asyncio
async def test_helper_wraps_block_shaped_args_before_mcp(monkeypatch):
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
    assert result["published"] is True
    spec = tool_calls[0][1]["spec"]
    assert spec["schema_version"] == 1
    assert spec["sections"][0]["blocks"][0]["text"] == "A claim"


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
                tool_calls=[{"name": "publish_canvas", "id": "call-1", "args": {"spec": {"schema_version": 1, "title": "X", "sections": []}}}],
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
