from dataclasses import asdict
from pathlib import Path

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
