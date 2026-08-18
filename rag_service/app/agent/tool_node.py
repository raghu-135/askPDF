from __future__ import annotations

from typing import Any

from langgraph.prebuilt import ToolNode


def format_recoverable_tool_error(error: Exception) -> str:
    return (
        f"Tool execution failed: {type(error).__name__}: {error}. "
        "Treat this source as unavailable and continue with other available evidence. "
        "If this source is required to answer, explain the limitation."
    )


class RecoverableToolNode(ToolNode):
    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("handle_tool_errors", format_recoverable_tool_error)
        super().__init__(*args, **kwargs)
