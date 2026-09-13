from __future__ import annotations

import pytest

from hermes_runtime.profile_manager import configured_mcp_url


def test_configured_mcp_url_projects_runtime_endpoint(monkeypatch):
    monkeypatch.setenv("ASKPDF_MCP_URL", "http://mcp.internal:8000/internal/mcp/")
    assert configured_mcp_url("offline") == "http://mcp.internal:8000/internal/hermes-mcp/offline/"
    assert configured_mcp_url("external") == "http://mcp.internal:8000/internal/hermes-mcp/external/"


def test_configured_mcp_url_rejects_missing_or_invalid(monkeypatch):
    monkeypatch.delenv("ASKPDF_MCP_URL", raising=False)
    with pytest.raises(RuntimeError):
        configured_mcp_url("offline")
    monkeypatch.setenv("ASKPDF_MCP_URL", "not-an-url")
    with pytest.raises(RuntimeError):
        configured_mcp_url("offline")
