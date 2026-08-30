"""Contract tests for the extracted dependency-free runtime protocol."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from app.runtime.contracts import AgentRuntimeRequest as CompatibilityRequest
from runtime_protocol.contracts import AgentRuntimeEvent, AgentRuntimeRequest
from runtime_protocol.errors import ProtocolDecodeError, ProtocolVersionError
from runtime_protocol.serialization import event_from_dict, iter_sse, request_from_dict


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("ASKPDF_REPO_DIR", ROOT.parent))
PROTOCOL_ROOT = ROOT / "runtime_protocol"
STDLIB_IMPORTS = {
    "__future__",
    "dataclasses",
    "json",
    "typing",
}


def test_compatibility_modules_reexport_the_shared_types():
    assert CompatibilityRequest is AgentRuntimeRequest


def test_protocol_package_has_only_standard_library_imports():
    for path in PROTOCOL_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = {
            node.module.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imports.update(
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert imports <= STDLIB_IMPORTS | {"runtime_protocol"}, path.name


def test_explicit_unsupported_contract_version_is_rejected():
    with pytest.raises(ProtocolVersionError, match="unsupported runtime contract version 2"):
        request_from_dict({
            "contract_version": 2,
            "run_id": "run-1",
            "thread_id": "thread-1",
            "definition_id": "definition-1",
            "framework": "langgraph",
            "builder_id": "langgraph_graph",
        })


def test_missing_required_request_field_has_a_protocol_error():
    with pytest.raises(ProtocolDecodeError, match="invalid runtime request"):
        request_from_dict({"run_id": "run-1"})


class _MalformedSseResponse:
    async def aiter_lines(self):
        for line in ("id: event-1", "event: run.failed", "data: {not-json", ""):
            yield line


@pytest.mark.asyncio
async def test_malformed_sse_data_has_a_protocol_error():
    with pytest.raises(ProtocolDecodeError, match="not valid JSON"):
        async for _ in iter_sse(_MalformedSseResponse()):
            pass


def test_event_round_trip_uses_the_shared_contract():
    event = AgentRuntimeEvent(
        event_id="event-1",
        run_id="run-1",
        sequence=1,
        kind="run.started",
    )
    assert event_from_dict(event.to_dict()) == event


def test_runtime_images_copy_the_shared_package():
    expected = {
        ROOT / "Dockerfile": "COPY runtime_protocol runtime_protocol",
        ROOT / "Dockerfile.test": "COPY runtime_protocol runtime_protocol",
        ROOT / "runtime_service" / "Dockerfile": "COPY runtime_protocol runtime_protocol",
        REPO_ROOT / "hermes_runtime" / "Dockerfile": "COPY rag_service/runtime_protocol runtime_protocol",
    }
    for path, copy_instruction in expected.items():
        assert copy_instruction in path.read_text(), str(path)


def test_runtime_services_import_the_shared_serializer():
    langgraph_source = (ROOT / "runtime_service" / "api.py").read_text()
    hermes_source = (REPO_ROOT / "hermes_runtime" / "api.py").read_text()
    assert "from runtime_protocol.serialization import" in langgraph_source
    assert "from runtime_protocol.serialization import" in hermes_source
