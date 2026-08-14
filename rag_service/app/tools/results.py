"""Compatibility import for the canonical askPDF tool result contract.

New MCP and framework-neutral code must import ``ToolResult`` from
``app.agent.tool_contract``.  This module remains temporarily so older provider
helpers do not create a second result model during the migration.
"""

from app.agent.tool_contract import ToolResult

__all__ = ["ToolResult"]
