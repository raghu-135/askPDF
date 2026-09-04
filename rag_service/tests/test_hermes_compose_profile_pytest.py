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
    assert adapter["depends_on"]["hermes"]["condition"] == "service_healthy"
    assert "COMPOSE_PROFILES" not in services["rag-service"].get("environment", {})
    assert services["rag-service"]["env_file"][0]["path"] == ".env"


def test_hermes_bootstrap_has_explicit_complete_environment():
    services = _compose("docker-compose.yml")["services"]
    bootstrap = services["hermes-config-init"]
    assert bootstrap.get("env_file") == []
    assert {
        "HERMES_DATA_ROOT", "HERMES_CONFIG_TEMPLATE_ROOT", "HERMES_MODEL_PROVIDER",
        "HERMES_MODEL_CONTEXT_LENGTH", "HERMES_PROFILE_ROOT", "HERMES_PROFILE_UID",
        "HERMES_PROFILE_GID", "API_SERVER_KEY", "HERMES_MCP_CONTEXT_SECRET",
        "OPENAI_API_KEY",
    } <= {entry.split("=", 1)[0] for entry in bootstrap["environment"]}


def test_runtime_integration_bootstrap_allowlists_provider_credential():
    bootstrap = _compose("docker-compose.runtime-integration.yml")["services"]["hermes-config-init"]
    assert "OPENAI_API_KEY" in bootstrap["environment"]
    assert bootstrap["environment"]["HERMES_MODEL_PROVIDER"] == "lmstudio"


def test_pinned_contract_copies_match_authoritative_module():
    from app.runtime.hermes_pinned_contract import HERMES_CONFIG_SCHEMA_VERSION, HERMES_REVISION

    root = REPOSITORY_ROOT
    assert HERMES_REVISION in (root / "docker-compose.yml").read_text()
    assert HERMES_REVISION in (root / "docker-compose.hermes-smoke.yml").read_text()
    assert HERMES_REVISION in (root / "hermes_fake/fixtures/run_events.json").read_text()
    assert f"_config_version: {HERMES_CONFIG_SCHEMA_VERSION}" in (root / "hermes_runtime/config.yaml").read_text()


def test_control_plane_and_gateway_pinned_contracts_are_identical():
    from app.runtime import hermes_pinned_contract as control_plane
    from hermes_runtime import pinned_contract as gateway

    def exported(module):
        return {
            name: value
            for name, value in vars(module).items()
            if name.startswith("HERMES_")
        }

    assert exported(control_plane) == exported(gateway)


def test_runtime_integration_compose_uses_the_same_pinned_real_hermes():
    compose = _compose("docker-compose.runtime-integration.yml")
    assert "hermes-fake" not in compose["services"]
    assert compose["services"]["hermes"]["build"]["context"].endswith("#bdd0a79c6a0ebc2344d5d6913c70bd89fa59c894")
    assert compose["services"]["hermes"]["environment"]["ASKPDF_HERMES_COMPAT_ENABLED"] == "1"
    assert "./hermes_runtime/hermes_pinned_patch:/opt/askpdf-hermes-pinned-patch:ro" in compose["services"]["hermes"]["volumes"]
    service = compose["services"]["hermes-runtime"]
    assert service["environment"]["HERMES_API_URL"] == "http://hermes:8642"
