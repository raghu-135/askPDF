"""Guard the neutral control-plane contract from framework imports."""

import ast
from pathlib import Path


NEUTRAL_MODULES = (
    "app/runtime/contracts.py",
    "app/runtime/errors.py",
    "app/runtime/transport.py",
    "app/runtime/adapter.py",
    "app/runtime/catalog.py",
    "app/runtime/builder.py",
    "app/runtime/http_adapter.py",
    "app/runtime/http_runtime_adapter.py",
    "app/runtime/hermes_adapter.py",
)
FORBIDDEN = ("langgraph", "langchain_core.messages", "langchain_core.tools")
ROOT = Path(__file__).resolve().parents[1]


def test_neutral_runtime_contract_modules_have_no_framework_imports():
    for relative in NEUTRAL_MODULES:
        tree = ast.parse((ROOT / relative).read_text(), filename=relative)
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(any(name == token or name.startswith(token + ".") for token in FORBIDDEN) for name in imported), relative


def test_control_plane_requirements_do_not_install_langgraph_checkpoint_packages():
    requirements = (ROOT / "requirements-control-plane.txt").read_text().lower()
    assert "langgraph" not in requirements
    assert "checkpoint-postgres" not in requirements


def test_langgraph_package_init_does_not_eagerly_import_framework_modules():
    source = (ROOT / "app/runtime/langgraph/__init__.py").read_text()
    tree = ast.parse(source, filename="app/runtime/langgraph/__init__.py")
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imported.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert "app.runtime.langgraph.compiler" not in imported
    assert "app.runtime.langgraph.graph" not in imported
    assert "app.runtime.langgraph.router_runtime" not in imported


def test_external_runtime_maintenance_does_not_import_local_checkpointing():
    source = (ROOT / "app/services/agent_task_maintenance.py").read_text()
    assert "if not _external_runtime_enabled():" in source


def test_external_langgraph_resolution_does_not_import_local_graph():
    source = (ROOT / "app/runtime/langgraph_builder.py").read_text()
    assert "if self._external_runtime_enabled():" in source
    assert "control plane only freezes neutral workflow inputs" in source
    assert "Materialization is framework-owned" in source
