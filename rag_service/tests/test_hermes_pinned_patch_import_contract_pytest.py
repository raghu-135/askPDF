from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SITECUSTOMIZE = REPOSITORY_ROOT / "hermes_runtime/hermes_pinned_patch/sitecustomize.py"


def test_pinned_sitecustomize_targets_v023_mcp_modules():
    text = SITECUSTOMIZE.read_text()
    assert "from tools.mcp_tool_config import _load_mcp_config" in text
    assert "from tools.mcp_tool_discovery import register_mcp_servers" in text
    assert "from tools import mcp_tool_handlers" in text
    assert "from tools.mcp_tool import _load_mcp_config" not in text
    assert "mcp_tool_handlers._make_tool_handler" in text


def test_pinned_sitecustomize_uses_scoped_mcp_teardown():
    text = SITECUSTOMIZE.read_text()
    assert "shutdown_mcp_servers(scope=scope" in text
    assert "_resolve_server_key" not in text
