"""Static guards for the external-only framework boundary."""

import ast
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(
    os.getenv("ASKPDF_REPO_DIR", str(ROOT.parent))
).resolve()
ROOT = REPOSITORY_ROOT / "rag_service"


def _source_files(relative_path: str, *suffixes: str) -> list[Path]:
    directory = REPOSITORY_ROOT / relative_path
    if not directory.is_dir() and relative_path == "runtime_protocol":
        spec = importlib.util.find_spec("runtime_protocol")
        if spec is not None and spec.submodule_search_locations:
            directory = Path(next(iter(spec.submodule_search_locations)))
    assert directory.is_dir(), f"expected source directory does not exist: {directory}"
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in suffixes
    )
    assert files, f"expected source files under {directory}"
    return files


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


def _dynamic_framework_references(path: Path) -> list[str]:
    """Catch string-based runtime imports that AST import checks cannot see."""
    source = path.read_text()
    return re.findall(
        r"(?:monkeypatch\.setattr|patch(?:\.object)?|import_module)\(\s*['\"]"
        r"(?:langgraph|langgraph_runtime)(?:\.|['\"])",
        source,
    )


def test_control_plane_has_no_langgraph_or_runtime_imports():
    forbidden = ("langgraph", "langgraph_runtime", "langchain_core.tools", "langchain_core.runnables")
    for path in _source_files("rag_service/app", ".py"):
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in _imports(path) for prefix in forbidden
        ), path.relative_to(ROOT)


def test_control_plane_tests_have_no_framework_execution_imports():
    forbidden = ("langgraph", "langgraph_runtime", "langchain_core.tools", "langchain_core.runnables")
    for path in _source_files("rag_service/tests", ".py"):
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in _imports(path) for prefix in forbidden
        ), path
        assert _dynamic_framework_references(path) == [], path


def test_runtime_has_no_control_plane_imports():
    for path in [*_source_files("langgraph_runtime", ".py"), *_source_files("hermes_runtime", ".py")]:
        assert not any(name == "app" or name.startswith("app.") for name in _imports(path)), path


def test_runtime_protocol_is_dependency_neutral():
    forbidden = ("app", "langgraph", "langchain", "sqlalchemy", "sqlmodel")
    for path in _source_files("runtime_protocol", ".py"):
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in _imports(path) for prefix in forbidden
        ), path.relative_to(ROOT)


def test_control_plane_manifest_and_legacy_paths_are_clean():
    requirements = (ROOT / "requirements.txt").read_text().lower()
    assert "langgraph" not in requirements
    assert "checkpoint-postgres" not in requirements
    assert not (ROOT / "app/runtime/mode.py").exists()
    assert not (ROOT / "app/runtime/langgraph_adapter.py").exists()
    assert not (ROOT / "app/runtime/langgraph").exists()
    assert not (ROOT / "runtime_service").exists()
    # Empty bind-mounted directories also create importable namespaces.
    assert not (ROOT / "langgraph_runtime").exists()
    # The integration test harness can inspect sibling sources via PYTHONPATH;
    # production/dev services expose only the control-plane source root.
    isolation_check = (
        "assert importlib.util.find_spec('langgraph') is None; "
        "assert importlib.util.find_spec('langgraph_runtime') is None; "
    ) if os.getenv("ASKPDF_ENFORCE_DEPENDENCY_ISOLATION") == "1" else ""
    subprocess.run([
        sys.executable, "-I", "-c",
        f"import sys, importlib.util; sys.path.insert(0, {str(REPOSITORY_ROOT)!r}); "
        + isolation_check
        + "assert importlib.util.find_spec('runtime_protocol') is not None",
    ], check=True)
    for legacy in (
        "product_orchestration/evidence.py",
        "product_orchestration/parallel_contracts.py",
        "product_orchestration/graph_validation.py",
        "product_orchestration/node_catalog.py",
        "product_orchestration/validator.py",
        "mcp/langchain_adapter.py",
    ):
        assert not (ROOT / "app" / legacy).exists()
    assert not (ROOT / "agent_workflows").exists()
    assert not (ROOT / "runtime_protocol").exists()


def test_control_plane_test_inventory_assigns_every_backend_test_file():
    from scripts.run_tests import (
        _approved_test_exclusions,
        _declared_control_plane_test_names,
    )

    repository_tests = {path.name for path in (ROOT / "tests").glob("test_*.py")}
    assigned = _declared_control_plane_test_names()
    excluded = set(_approved_test_exclusions())
    assert not assigned & excluded
    assert repository_tests == assigned | excluded


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
        *_source_files("rag_service/app", ".py"),
        *_source_files("frontend/src", ".ts", ".tsx"),
    ]
    offenders = [
        path for path in product_sources
        if "checkpoint_thread_id" in path.read_text()
    ]
    assert offenders == []
