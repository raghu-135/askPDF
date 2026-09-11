import os
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(
    os.getenv("ASKPDF_REPO_DIR", str(Path(__file__).resolve().parents[2]))
)


def _compose(name: str) -> dict:
    class ComposeLoader(yaml.SafeLoader):
        pass

    ComposeLoader.add_constructor("!reset", lambda loader, node: loader.construct_sequence(node))
    return yaml.load((REPOSITORY_ROOT / name).read_text(), Loader=ComposeLoader)


def test_example_environment_enables_hermes_with_distinct_placeholder_tokens():
    values = {}
    for line in (REPOSITORY_ROOT / ".env.example").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value

    assert values["COMPOSE_PROFILES"] == "hermes"
    assert values["HERMES_RUNTIME_TOKEN"] != values["HERMES_API_TOKEN"]
    assert values["HERMES_API_TOKEN"].startswith("replace-with-")


def test_bootstrap_profiles_defer_mcp_to_isolated_run_profiles():
    paths = [
        "hermes_runtime/config.yaml",
        "hermes_runtime/profiles/askpdf-deep-offline/config.yaml",
        "hermes_runtime/profiles/askpdf-deep-external/config.yaml",
    ]
    for relative_path in paths:
        config = yaml.safe_load((REPOSITORY_ROOT / relative_path).read_text())
        assert config["mcp_servers"] == {}
        assert config["auxiliary"]["title_generation"]["enabled"] is False


def test_main_compose_keeps_pinned_real_hermes_opt_in():
    services = _compose("docker-compose.yml")["services"]
    hermes = services["hermes"]
    adapter = services["hermes-runtime"]
    assert hermes["profiles"] == ["hermes"]
    assert adapter["profiles"] == ["hermes"]
    assert services["hermes-config-init"]["profiles"] == ["hermes"]
    assert "${HERMES_UPSTREAM_REVISION:?" in hermes["build"]["context"]
    assert hermes["healthcheck"]["test"][-1].endswith("/health")
    assert "ASKPDF_HERMES_COMPAT_ENABLED=1" in set(hermes["environment"])
    assert "./hermes_runtime/hermes_pinned_patch:/opt/askpdf-hermes-pinned-patch:ro" in hermes["volumes"]
    assert adapter["healthcheck"]["test"][-1].endswith("/readyz")
    assert "HERMES_API_URL=http://hermes:8642" in set(adapter["environment"])
    assert any(
        entry.startswith("HERMES_RUNTIME_WORKERS=")
        for entry in adapter["environment"]
    )
    assert adapter["depends_on"]["hermes"]["condition"] == "service_healthy"
    assert "COMPOSE_PROFILES" not in services["rag-service"].get("environment", {})
    assert services["rag-service"]["env_file"][0]["path"] == ".env"


def test_dev_hermes_runtime_does_not_inherit_control_plane_mcp_transport():
    services = _compose("docker-compose.dev.yml")["services"]
    assert services["hermes-runtime"]["environment"]["MCP_TRANSPORT"] == "loopback_http"
    assert services["hermes-runtime"]["environment"]["MCP_LOOPBACK_URL"] == "http://rag-service:8000/internal/mcp/"


def test_hermes_bootstrap_has_explicit_complete_environment():
    services = _compose("docker-compose.yml")["services"]
    bootstrap = services["hermes-config-init"]
    assert bootstrap.get("env_file") == []
    assert {
        "HERMES_DATA_ROOT", "HERMES_CONFIG_TEMPLATE_ROOT", "HERMES_MODEL_PROVIDER",
        "HERMES_MODEL_CONTEXT_LENGTH", "HERMES_PROFILE_ROOT", "HERMES_PROFILE_UID",
        "HERMES_PROFILE_GID", "HERMES_API_TOKEN",
        "OPENAI_API_KEY",
    } <= {entry.split("=", 1)[0] for entry in bootstrap["environment"]}


def test_runtime_integration_bootstrap_allowlists_provider_credential():
    bootstrap = _compose("docker-compose.runtime-integration.yml")["services"]["hermes-config-init"]
    assert "OPENAI_API_KEY" in bootstrap["environment"]
    assert bootstrap["environment"]["HERMES_MODEL_PROVIDER"] == "lmstudio"


def test_pinned_contract_copies_match_authoritative_module():
    from runtime_protocol.hermes_contract import HERMES_CONFIG_SCHEMA_VERSION, HERMES_REVISION

    root = REPOSITORY_ROOT
    assert HERMES_REVISION in (root / "docker-compose.yml").read_text()
    assert HERMES_REVISION in (root / "docker-compose.hermes-smoke.yml").read_text()
    assert HERMES_REVISION in (root / "hermes_fake/fixtures/run_events.json").read_text()
    assert f"_config_version: {HERMES_CONFIG_SCHEMA_VERSION}" in (root / "hermes_runtime/config.yaml").read_text()


def test_control_plane_and_gateway_use_one_pinned_contract():
    from runtime_protocol.hermes_contract import HERMES_OFFLINE_PROFILE, HERMES_PROFILE_NAMES
    from app.runtime import hermes_profile
    from hermes_runtime import profile_manager

    assert hermes_profile.HERMES_OFFLINE_PROFILE == HERMES_OFFLINE_PROFILE
    assert profile_manager.HERMES_PROFILE_NAMES is HERMES_PROFILE_NAMES
    assert not (REPOSITORY_ROOT / "hermes_runtime/pinned_contract.py").exists()
    assert not (REPOSITORY_ROOT / "rag_service/app/runtime/hermes_pinned_contract.py").exists()


def test_runtime_integration_compose_uses_the_same_pinned_real_hermes():
    compose = _compose("docker-compose.runtime-integration.yml")
    assert "hermes-fake" not in compose["services"]
    assert compose["services"]["hermes"]["build"]["context"].endswith("#bdd0a79c6a0ebc2344d5d6913c70bd89fa59c894")
    assert compose["services"]["hermes"]["environment"]["ASKPDF_HERMES_COMPAT_ENABLED"] == "1"
    assert "./hermes_runtime/hermes_pinned_patch:/opt/askpdf-hermes-pinned-patch:ro" in compose["services"]["hermes"]["volumes"]
    assert compose["services"]["hermes"]["depends_on"]["rag-service"]["condition"] == "service_healthy"
    control_plane = compose["services"]["rag-service"]
    assert control_plane["environment"]["HERMES_RUNTIME_URL"] == "http://hermes-runtime:8200"
    assert control_plane["environment"]["ASKPDF_MCP_REQUIRED"] == "true"
    service = compose["services"]["hermes-runtime"]
    assert service["environment"]["HERMES_API_URL"] == "http://hermes:8642"
    assert service["environment"]["ASKPDF_MCP_REQUIRED"] == "true"
    assert service["depends_on"]["rag-service"]["condition"] == "service_healthy"
    assert service["healthcheck"]["test"][-1].endswith("/readyz")
    assert "HERMES_RUNTIME_TOKEN" in service["environment"]
    assert "API_SERVER_KEY" not in service["environment"]


def test_control_plane_test_runner_does_not_inherit_ci_loopback_mcp():
    environment = _compose("docker-compose.test.yml")["services"]["test-runner"]["environment"]
    assert "MCP_TRANSPORT=in_process" in environment
    assert "PYTHONPATH=/workspace:/app" in environment
    assert "MCP_LOOPBACK_URL=http://127.0.0.1:8000/internal/mcp/" in environment
    assert "ASKPDF_MCP_URL=http://127.0.0.1:8000/internal/mcp/" in environment


def test_ci_environment_includes_hermes_mcp_urls():
    values = {}
    for line in (REPOSITORY_ROOT / ".env.ci").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    assert values["MCP_TRANSPORT"] == "loopback_http"
    assert values["ASKPDF_MCP_URL"] == "http://rag-service:8000/internal/mcp/"
    assert values["ASKPDF_MCP_HEALTH_URL"] == "http://rag-service:8000/health"
    assert values["HERMES_API_URL"] == "http://hermes:8642"


def test_ci_environment_includes_control_plane_tool_instruction_limit():
    values = {}
    for line in (REPOSITORY_ROOT / ".env.ci").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value

    assert values["MAX_TOOL_INSTRUCTION_CHARS"] == "500"


def test_main_compose_does_not_mount_the_project_environment_into_hermes():
    hermes = _compose("docker-compose.yml")["services"]["hermes"]
    assert all(".env:" not in volume for volume in hermes.get("volumes", []))
