"""Static guards for the external-only framework boundary."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    values = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    values.update(
        alias.name for node in ast.walk(tree)
        if isinstance(node, ast.Import) for alias in node.names
    )
    return values


def test_control_plane_has_no_langgraph_or_runtime_imports():
    forbidden = ("langgraph", "langgraph_runtime", "langchain_core.tools", "langchain_core.runnables")
    for path in (ROOT / "app").rglob("*.py"):
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in _imports(path) for prefix in forbidden
        ), path.relative_to(ROOT)


def test_runtime_has_no_control_plane_imports():
    for path in (REPOSITORY_ROOT / "langgraph_runtime").rglob("*.py"):
        assert not any(name == "app" or name.startswith("app.") for name in _imports(path)), path


def test_runtime_protocol_is_dependency_neutral():
    forbidden = ("app", "langgraph", "langchain", "sqlalchemy", "sqlmodel")
    for path in (ROOT / "runtime_protocol").rglob("*.py"):
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in _imports(path) for prefix in forbidden
        ), path.relative_to(ROOT)


def test_control_plane_manifest_and_legacy_paths_are_clean():
    requirements = (ROOT / "requirements-control-plane.txt").read_text().lower()
    assert "langgraph" not in requirements
    assert "checkpoint-postgres" not in requirements
    assert not (ROOT / "app/runtime/mode.py").exists()
    assert not (ROOT / "app/runtime/langgraph_adapter.py").exists()
    assert not (ROOT / "app/runtime/langgraph").exists()
    assert not (ROOT / "runtime_service").exists()
    for legacy in (
        "agent_workflows/evidence.py",
        "agent_workflows/parallel_contracts.py",
        "agent_workflows/graph_validation.py",
        "agent_workflows/node_catalog.py",
        "agent_workflows/validator.py",
        "mcp/langchain_adapter.py",
    ):
        assert not (ROOT / "app" / legacy).exists()


def test_control_plane_has_no_framework_execution_symbols():
    forbidden = ("StateGraph", "RunnableConfig", "GraphInterrupt", "RuntimeExecutionContext")
    offenders = {
        str(path.relative_to(ROOT)): token
        for path in (ROOT / "app").rglob("*.py")
        for token in forbidden
        if token in path.read_text()
    }
    assert offenders == {}


def test_frontend_and_product_api_do_not_expose_checkpoint_identity():
    product_sources = [
        *(ROOT / "app").rglob("*.py"),
        *(REPOSITORY_ROOT / "frontend/src").rglob("*.ts"),
        *(REPOSITORY_ROOT / "frontend/src").rglob("*.tsx"),
    ]
    offenders = [
        path for path in product_sources
        if "checkpoint_thread_id" in path.read_text() and path.name != "models_sqlmodel.py"
    ]
    assert offenders == []
