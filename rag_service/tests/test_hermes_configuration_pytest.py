import pytest

from app.runtime.hermes_config import (
    HermesConfigurationError,
    hermes_model_context_length,
    validate_hermes_model_compatibility,
)


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
