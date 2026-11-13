import os
from pathlib import Path


REPOSITORY_ROOT = Path(os.getenv("ASKPDF_REPO_DIR", str(Path(__file__).resolve().parents[2])))
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


def test_pinned_sitecustomize_exports_openrouter_provider_cost():
    text = SITECUSTOMIZE.read_text()
    assert "merge_openrouter_usage_request" in text
    assert "api_server_runs._run_usage" in text
    assert "turn_usage.record_response_usage" in text
    assert "turn_response_check.record_response_usage" in text
    assert "run_turn_runner._settle_api_runtime_lock" in text


def test_pinned_sitecustomize_installs_without_gateway_argv_guard():
    text = SITECUSTOMIZE.read_text()
    assert '"gateway" in sys.argv' not in text
