import os
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(
    os.getenv("ASKPDF_REPO_DIR", str(Path(__file__).resolve().parents[2]))
)


def _compose(name: str) -> dict:
    return yaml.safe_load((REPOSITORY_ROOT / name).read_text())


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


def test_main_compose_runs_the_pinned_real_hermes_by_default():
    services = _compose("docker-compose.yml")["services"]
    hermes = services["hermes"]
    adapter = services["hermes-runtime"]
    assert "profiles" not in hermes
    assert "profiles" not in adapter
    assert hermes["build"]["context"].endswith("#bdd0a79c6a0ebc2344d5d6913c70bd89fa59c894")
    assert hermes["healthcheck"]["test"][-1].endswith("/health")
    assert "ASKPDF_HERMES_COMPAT_ENABLED=1" in set(hermes["environment"])
    assert "./hermes_runtime/hermes_compat:/opt/askpdf-hermes-compat:ro" in hermes["volumes"]
    assert adapter["healthcheck"]["test"][-1].endswith("/readyz")
    assert "HERMES_API_URL=http://hermes:8642" in set(adapter["environment"])
    assert adapter["depends_on"]["hermes"]["condition"] == "service_healthy"


def test_runtime_integration_compose_uses_the_same_pinned_real_hermes():
    compose = _compose("docker-compose.runtime-integration.yml")
    assert "hermes-fake" not in compose["services"]
    assert compose["services"]["hermes"]["build"]["context"].endswith("#bdd0a79c6a0ebc2344d5d6913c70bd89fa59c894")
    assert compose["services"]["hermes"]["environment"]["ASKPDF_HERMES_COMPAT_ENABLED"] == "1"
    assert "./hermes_runtime/hermes_compat:/opt/askpdf-hermes-compat:ro" in compose["services"]["hermes"]["volumes"]
    service = compose["services"]["hermes-runtime"]
    assert service["environment"]["HERMES_API_URL"] == "http://hermes:8642"
