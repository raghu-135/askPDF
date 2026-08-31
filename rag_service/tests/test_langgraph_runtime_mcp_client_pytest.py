import json

import pytest


@pytest.mark.asyncio
async def test_runtime_mcp_wrapper_preserves_search_arguments(monkeypatch):
    from langgraph_runtime import mcp_client

    calls = []

    async def fake_call(name, arguments, config):
        calls.append((name, arguments, config))
        return json.dumps({"ok": True, "content": "ok"})

    monkeypatch.setattr(mcp_client, "_call", fake_call)
    tool = mcp_client.create_mcp_langchain_tool("search_documents")
    await tool.ainvoke(
        {"query": "external runtime boundary", "max_results": 7},
        config={"configurable": {"mcp_execution_context_token": "grant"}},
    )

    assert calls[0][0] == "search_documents"
    assert calls[0][1] == {"query": "external runtime boundary", "max_results": 7}


def test_runtime_mcp_wrapper_advertises_required_query_schema():
    from langgraph_runtime.mcp_client import create_mcp_langchain_tool

    schema = create_mcp_langchain_tool("search_web").args_schema.model_json_schema()
    assert schema["required"] == ["query"]

