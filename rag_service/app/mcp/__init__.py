"""First-party MCP protocol boundary for rag-service tools.

Keep the server import lazy so the external runtime can use the loopback MCP
client without importing rag-service's database-backed tool registry.
"""

__all__ = ["MCPServer"]


def __getattr__(name: str):
    if name == "MCPServer":
        from app.mcp.server import MCPServer

        return MCPServer
    raise AttributeError(name)
