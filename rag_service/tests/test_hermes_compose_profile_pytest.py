import os
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(
    os.getenv("ASKPDF_REPO_DIR", str(Path(__file__).resolve().parents[2]))
)


def _compose(name: str) -> dict:
    return yaml.safe_load((REPOSITORY_ROOT / name).read_text())


def test_main_compose_keeps_hermes_in_proof_profile_and_checks_readiness():
    service = _compose("docker-compose.yml")["services"]["hermes-runtime"]
    assert service["profiles"] == ["second-runtime-proof"]
    assert service["healthcheck"]["test"][-1].endswith("/readyz")
    environment = set(service["environment"])
    assert "HERMES_API_URL=${HERMES_API_URL:-}" in environment
    assert not any("http://hermes-agent" in value for value in environment)


def test_runtime_integration_compose_uses_explicit_fake_hermes_upstream():
    compose = _compose("docker-compose.runtime-integration.yml")
    service = compose["services"]["hermes-runtime"]
    assert service["environment"]["HERMES_API_URL"] == "http://hermes-fake:8000"
    assert "second-runtime-proof" not in service.get("profiles", [])
