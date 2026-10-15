from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import ValidationError

from langgraph_runtime.mcp_client import create_mcp_langchain_tool
from langgraph_runtime.workflows.canvas_publish import (
    CanvasSpecNormalizeError,
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
        "title": "Sources",
        "citations": [{"kind": "web", "label": "Paper", "url": "https://example.com"}],
    }


def test_normalize_publish_canvas_spec_preserves_canvas_spec_v1():
    original = {"schema_version": 1, "title": "Advisor Models", "sections": [{"title": "Coverage", "blocks": [{"type": "stat", "value": "3", "label": "Claims"}]}]}
    assert normalize_publish_canvas_spec({"spec": original}) == original


def test_normalize_publish_canvas_spec_strips_unknown_envelope_fields():
    spec = normalize_publish_canvas_spec({
        "spec": {
            "schema_version": 1,
            "title": "How Humans Learn",
            "id": "artifact-uuid",
            "subtitle": "Report overview",
            "version": 2,
            "sections": [{
                "title": "Overview",
                "blocks": [{"type": "markdown", "text": "Summary.", "extra": "drop-me"}],
            }],
        },
    })
    assert set(spec.keys()) == {"schema_version", "title", "sections"}
    assert "id" not in spec
    assert "subtitle" not in spec
    assert "version" not in spec
    assert spec["sections"][0]["blocks"][0] == {"type": "markdown", "text": "Summary."}


def test_normalize_publish_canvas_spec_defaults_missing_title_and_schema_version():
    spec = normalize_publish_canvas_spec({
        "spec": {
            "sections": [{
                "blocks": [{"type": "markdown", "text": "Learning happens through practice."}],
            }],
        },
    })
    assert spec["schema_version"] == 1
    assert spec["title"] == "Findings"
    assert spec["sections"][0]["title"] == "Findings"
    assert spec["sections"][0]["blocks"][0]["text"] == "Learning happens through practice."


def test_normalize_publish_canvas_spec_repairs_incomplete_blocks_and_citations():
    spec = normalize_publish_canvas_spec({
        "spec": {
            "sections": [{
                "title": "Overview",
                "blocks": [
                    {"type": "callout", "tone": "warning", "content": "Practice builds durable memory."},
                    {"type": "markdown", "content": "Learning is active."},
                    {
                        "type": "sources",
                        "citations": [{
                            "kind": "document",
                            "label": "How Humans Learn",
                            "source_id": "abc123def456",
                        }],
                    },
                ],
            }],
        },
    })
    types = [block["type"] for block in spec["sections"][0]["blocks"]]
    assert "callout" in types and "markdown" in types
    callout = next(block for block in spec["sections"][0]["blocks"] if block["type"] == "callout")
    assert callout["title"] == "Note"
    assert callout["body"] == "Practice builds durable memory."
    sources = next(block for block in spec["sections"][0]["blocks"] if block["type"] == "sources")
    assert sources["citations"][0]["file_hash"] == "abc123def456"
    assert "source_id" not in sources["citations"][0]


def test_normalize_publish_canvas_spec_wraps_string_sections():
    spec = normalize_publish_canvas_spec({
        "spec": {"sections": ["1. Attach prior knowledge before new concepts."]},
    })
    assert spec["sections"][0]["blocks"][0]["text"].startswith("1. Attach")


def test_normalize_publish_canvas_spec_uses_subtitle_as_title_when_title_missing():
    spec = normalize_publish_canvas_spec({
        "spec": {
            "subtitle": "How Humans Learn",
            "sections": [{"title": "Overview", "blocks": [{"type": "stat", "value": "3", "label": "Ideas"}]}],
        },
    })
    assert spec["title"] == "How Humans Learn"
    assert "subtitle" not in spec


def test_normalize_publish_canvas_spec_preserves_dag_edge_relation():
    block = {
        "type": "dag",
        "nodes": [{"id": "a", "label": "Claim"}, {"id": "b", "label": "Source"}],
        "edges": [{"source": "a", "target": "b", "relation": "contradicts"}],
    }
    spec = normalize_publish_canvas_spec(block)
    edge = spec["sections"][0]["blocks"][0]["edges"][0]
    assert edge["relation"] == "contradicts"


def test_normalize_publish_canvas_spec_maps_dag_from_to_aliases():
    spec = normalize_publish_canvas_spec({
        "spec": {
            "schema_version": 1,
            "title": "Flow",
            "sections": [{
                "title": "Structure",
                "blocks": [{
                    "type": "dag",
                    "nodes": [{"id": "a", "title": "Start"}, {"id": "b", "name": "End"}],
                    "edges": [{"from": "a", "to": "b"}],
                }],
            }],
        },
    })
    block = spec["sections"][0]["blocks"][0]
    assert block["nodes"][0]["label"] == "Start"
    assert block["nodes"][1]["label"] == "End"
    assert block["edges"][0] == {"source": "a", "target": "b"}


def test_normalize_publish_canvas_spec_repairs_type_as_key_wrappers():
    """Matches evaluator_replanner publish_canvas payloads (markdown/stat/sources keys)."""
    spec = normalize_publish_canvas_spec({
        "spec": {
            "schema_version": 1,
            "title": "Five Core Ideas",
            "sections": [{
                "title": "Overview",
                "blocks": [
                    {"markdown": "Five core ideas from the report."},
                    {"stat": {"label": "Core Ideas Identified", "value": "5"}},
                    {
                        "sources": {
                            "citations": [{
                                "kind": "document",
                                "label": "how-humans-learn-report.pdf",
                                "file_hash": "255dc7dfd217005fef041fea229249b2",
                            }],
                        },
                    },
                ],
            }],
        },
    })
    blocks = spec["sections"][0]["blocks"]
    assert blocks[0] == {"type": "markdown", "text": "Five core ideas from the report."}
    assert blocks[1] == {"type": "stat", "label": "Core Ideas Identified", "value": "5"}
    assert blocks[2]["type"] == "sources"
    assert blocks[2]["citations"][0]["file_hash"] == "255dc7dfd217005fef041fea229249b2"


def test_normalize_publish_canvas_spec_drops_uncoercible_blocks_instead_of_passthrough():
    with pytest.raises(CanvasSpecNormalizeError, match="no publishable sections"):
        normalize_publish_canvas_spec({
            "spec": {
                "title": "Broken",
                "sections": [{"title": "Overview", "blocks": [{"not_a_block": {"x": 1}}]}],
            },
        })


def test_normalize_publish_canvas_spec_section_content_becomes_blocks():
    spec = normalize_publish_canvas_spec({
        "spec": {
            "title": "Learn",
            "sections": [{"title": "Overview", "content": "Humans learn through practice."}],
        },
    })
    assert spec["sections"][0]["blocks"] == [{"type": "markdown", "text": "Humans learn through practice."}]


def test_normalize_publish_canvas_spec_rejects_empty_sections():
    with pytest.raises(CanvasSpecNormalizeError, match="no publishable sections"):
        normalize_publish_canvas_spec({
            "spec": {"schema_version": 1, "title": "Empty", "sections": []},
        })


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
                    "args": {"spec": {"schema_version": 1, "title": "X", "sections": []}},
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

    spec = {
        "schema_version": 1,
        "title": "Advisor Models",
        "sections": [{"title": "Coverage", "blocks": [{"type": "stat", "value": "3", "label": "Claims"}]}],
    }

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
    assert "could not be published" in result["answer"]
