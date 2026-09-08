import pytest
from langchain_core.tools import tool

from langgraph_runtime.agent.tool_node import RecoverableToolNode


def test_orchestrator_tool_node_configures_recoverable_tool_errors():
    @tool
    def failing_tool(query: str) -> str:
        """Test tool that always fails."""
        raise RuntimeError("simulated tool outage")

    node = RecoverableToolNode([failing_tool])
    message = node._handle_tool_errors(RuntimeError("simulated tool outage"))

    assert "Tool execution failed: RuntimeError: simulated tool outage" in message
    assert "continue with other available evidence" in message
