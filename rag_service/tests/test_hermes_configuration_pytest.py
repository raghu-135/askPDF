import pytest

from app.runtime.hermes_config import (
    HermesConfigurationError,
    hermes_runtime_enabled,
    hermes_model_context_length,
    validate_hermes_model_compatibility,
)
from runtime_protocol.hermes_contract import HERMES_CHAT_PROVIDER


@pytest.mark.parametrize("profiles", ["hermes", "langgraph,hermes", " HERMES "])
def test_hermes_profile_is_the_single_enablement_switch(monkeypatch, profiles):
    monkeypatch.setenv("COMPOSE_PROFILES", profiles)
    assert hermes_runtime_enabled() is True


@pytest.mark.parametrize("profiles", ["", "langgraph", "real-hermes", "true"])
def test_hermes_is_disabled_without_exact_compose_profile(monkeypatch, profiles):
    monkeypatch.setenv("COMPOSE_PROFILES", profiles)
    assert hermes_runtime_enabled() is False


def test_hermes_chat_uses_the_shared_custom_provider():
    assert HERMES_CHAT_PROVIDER == "custom"


@pytest.mark.parametrize("configured", ["64000", "65536", "131072"])
def test_hermes_context_length_uses_exact_deployment_value(monkeypatch, configured):
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", configured)
    assert hermes_model_context_length() == int(configured)
    assert validate_hermes_model_compatibility() == int(configured)


@pytest.mark.parametrize("configured", [None, "", "true", "false", "2047", "32768", "8k"])
def test_hermes_context_length_rejects_missing_or_invalid_values(monkeypatch, configured):
    if configured is None:
        monkeypatch.delenv("HERMES_MODEL_CONTEXT_LENGTH", raising=False)
    else:
        monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", configured)
    with pytest.raises(HermesConfigurationError):
        hermes_model_context_length()
