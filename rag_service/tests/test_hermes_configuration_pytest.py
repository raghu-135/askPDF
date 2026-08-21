import pytest

from app.runtime.hermes_config import (
    HermesConfigurationError,
    hermes_runtime_enabled,
    hermes_model_context_length,
    validate_hermes_model_compatibility,
)
from app.runtime.hermes_compatibility import provider_requires_api_key


@pytest.mark.parametrize("profiles", ["hermes", "langgraph,hermes", " HERMES "])
def test_hermes_profile_is_the_single_enablement_switch(monkeypatch, profiles):
    monkeypatch.setenv("COMPOSE_PROFILES", profiles)
    assert hermes_runtime_enabled() is True


@pytest.mark.parametrize("profiles", ["", "langgraph", "real-hermes", "true"])
def test_hermes_is_disabled_without_exact_compose_profile(monkeypatch, profiles):
    monkeypatch.setenv("COMPOSE_PROFILES", profiles)
    assert hermes_runtime_enabled() is False


def test_lmstudio_is_the_pinned_keyless_provider():
    assert provider_requires_api_key("lmstudio") is False
    assert provider_requires_api_key(" LMSTUDIO ") is False
    assert provider_requires_api_key("custom") is True


@pytest.mark.parametrize("configured", ["8192", "32768", "131072"])
def test_hermes_context_length_uses_exact_deployment_value(monkeypatch, configured):
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", configured)
    monkeypatch.setenv("HERMES_MODEL_PROVIDER", "lmstudio")
    assert hermes_model_context_length() == int(configured)
    assert validate_hermes_model_compatibility() == (int(configured), "lmstudio")


@pytest.mark.parametrize("configured", [None, "", "true", "false", "2047", "8k"])
def test_hermes_context_length_rejects_missing_or_invalid_values(monkeypatch, configured):
    if configured is None:
        monkeypatch.delenv("HERMES_MODEL_CONTEXT_LENGTH", raising=False)
    else:
        monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", configured)
    with pytest.raises(HermesConfigurationError):
        hermes_model_context_length()


def test_non_lmstudio_provider_obeys_pinned_64k_floor(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "32768")
    monkeypatch.setenv("HERMES_MODEL_PROVIDER", "custom")
    with pytest.raises(HermesConfigurationError) as exc_info:
        validate_hermes_model_compatibility()
    assert exc_info.value.code == "hermes_context_length_provider_incompatible"
