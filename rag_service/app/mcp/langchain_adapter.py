"""LangChain tool wrappers over the framework-neutral MCP boundary."""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from app.agent.tool_registry import TOOL_FRIENDLY_CONFIG
from app.mcp.discovery import request_model_for_tool
from app.mcp.tool_adapter import _arguments_from_input, call_mcp_tool


def create_mcp_langchain_tool(
    tool_name: str,
    request_model: type[Any] | None = None,
) -> BaseTool:
    """Create a LangChain-compatible wrapper backed exclusively by MCP."""
    model = request_model or request_model_for_tool(tool_name)

    async def invoke(
        *args: Any,
        config: RunnableConfig = None,
        **kwargs: Any,
    ) -> str:
        arguments = dict(kwargs)
        if args:
            arguments.update(_arguments_from_input(args[0]))
        return await call_mcp_tool(tool_name, arguments, config)

    config = TOOL_FRIENDLY_CONFIG[tool_name]
    return StructuredTool.from_function(
        coroutine=invoke,
        name=tool_name,
        description=config["description"],
        args_schema=model,
    )


def create_wikipedia_tool() -> BaseTool:
    return create_mcp_langchain_tool("wikipedia")


def create_thread_shape_tool() -> BaseTool:
    return create_mcp_langchain_tool("get_thread_shape")
