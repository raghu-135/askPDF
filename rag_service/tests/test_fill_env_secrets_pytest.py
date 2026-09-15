from __future__ import annotations

import importlib.util
import os
from pathlib import Path

REPO = Path(os.getenv("ASKPDF_REPO_DIR", str(Path(__file__).resolve().parents[2])))


def _load_fill_env_secrets():
    path = REPO / "scripts" / "fill_env_secrets.py"
    spec = importlib.util.spec_from_file_location("fill_env_secrets", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fill_env_secrets_replaces_placeholders_once(tmp_path: Path):
    fill_env = _load_fill_env_secrets()
    env_path = tmp_path / ".env"
    env_path.write_text((REPO / ".env.example").read_text())

    first = fill_env.fill_env_file(env_path)
    assert first == [
        "LANGGRAPH_RUNTIME_TOKEN",
        "LANGGRAPH_RUNTIME_BINDING_SECRET",
        "MCP_EXECUTION_CONTEXT_SECRET",
        "ASKPDF_ADMIN_TOKEN",
        "HERMES_RUNTIME_TOKEN",
        "HERMES_API_TOKEN",
    ]
    values = {}
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    secrets = [values[name] for name in first]
    assert all(len(secret) >= 32 for secret in secrets)
    assert len(set(secrets)) == len(secrets)
    assert not any(
        secret.lower().startswith(fill_env._PLACEHOLDER_PREFIXES)
        for secret in secrets
    )
    assert values["OPENAI_API_KEY"] == ""
    assert fill_env.fill_env_file(env_path) == []


def test_fill_env_secrets_exports_runtime_file(tmp_path: Path):
    fill_env = _load_fill_env_secrets()
    env_path = tmp_path / ".env"
    env_path.write_text((REPO / ".env.example").read_text())
    export_path = tmp_path / "runtime.env"
    ready_path = tmp_path / "ready"

    secrets_map, filled, origin = fill_env.resolve_service_secrets(
        env_path,
        example_path=REPO / ".env.example",
        environ={},
    )
    fill_env.write_runtime_secrets_file(secrets_map, export_path)
    ready_path.write_text("ready\n")

    assert origin == "file"
    assert filled == [
        "LANGGRAPH_RUNTIME_TOKEN",
        "LANGGRAPH_RUNTIME_BINDING_SECRET",
        "MCP_EXECUTION_CONTEXT_SECRET",
        "ASKPDF_ADMIN_TOKEN",
        "HERMES_RUNTIME_TOKEN",
        "HERMES_API_TOKEN",
    ]
    exported = fill_env.parse_env_assignments(export_path.read_text())
    assert list(exported) == list(fill_env.SERVICE_SECRET_NAMES)
    assert exported == secrets_map
    assert ready_path.read_text() == "ready\n"


def test_fill_env_secrets_exports_process_environment_when_env_is_missing(tmp_path: Path):
    fill_env = _load_fill_env_secrets()
    env_path = tmp_path / ".env"
    environ = {name: f"{name.lower().replace('_', '-')}-value-32-characters-xx" for name in fill_env.SERVICE_SECRET_NAMES}
    secrets_map, filled, origin = fill_env.resolve_service_secrets(
        env_path,
        example_path=REPO / ".env.example",
        environ=environ,
    )
    assert origin == "environ"
    assert filled == []
    assert not env_path.exists()
    assert secrets_map["ASKPDF_ADMIN_TOKEN"] == environ["ASKPDF_ADMIN_TOKEN"]


def test_fill_env_secrets_copies_example_when_env_is_missing(tmp_path: Path):
    fill_env = _load_fill_env_secrets()
    env_path = tmp_path / ".env"
    filled = fill_env.fill_env_file(env_path, example_path=REPO / ".env.example")
    assert env_path.exists()
    assert "ASKPDF_ADMIN_TOKEN" in filled
