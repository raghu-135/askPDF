from dataclasses import asdict
from pathlib import Path
import importlib.util
import json
import pytest

from runtime_protocol.contracts import AgentDefinition, RuntimeOperationId
from runtime_protocol.transport import definition_from_dict


def test_shared_protocol_is_one_top_level_dependency_neutral_package():
    repository_root = Path(__file__).resolve().parents[2]
    package_path = Path(__import__("runtime_protocol").__file__).resolve().parent
    assert package_path.name == "runtime_protocol"
    if (repository_root / "runtime_protocol").is_dir():
        assert package_path == repository_root / "runtime_protocol"

    definition = AgentDefinition(
        definition_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
    )
    restored = definition_from_dict(asdict(definition))
    assert restored == definition
    assert RuntimeOperationId.RUN_START.value == "run.start"


@pytest.mark.parametrize("case", ["missing", "nonexistent", "duplicate", "overlap"])
def test_test_inventory_rejects_invalid_ownership(tmp_path, monkeypatch, case):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_tests.py"
    spec = importlib.util.spec_from_file_location("test_inventory_runner", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "APP_DIR", tmp_path)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_owned.py").write_text("")
    (tests / "test_inventory.json").write_text(json.dumps({"excluded": {"test_owned.py": "external proof"} if case == "overlap" else {}}))
    for name in ("UNIT_TEST_FILES", "HERMES_TEST_FILES", "MCP_TEST_FILES", "DB_TEST_FILES", "API_TEST_FILES", "INTEGRATION_TEST_FILES", "SCHEMA_TEST_FILES"):
        monkeypatch.setattr(runner, name, [])
    if case != "missing":
        monkeypatch.setattr(runner, "UNIT_TEST_FILES", ["test_owned.py"])
    if case == "nonexistent":
        monkeypatch.setattr(runner, "UNIT_TEST_FILES", ["test_owned.py", "test_absent.py"])
    if case == "duplicate":
        monkeypatch.setattr(runner, "HERMES_TEST_FILES", ["test_owned.py"])
    with pytest.raises(SystemExit, match="ownership|inventory"):
        runner._validate_test_inventory()
