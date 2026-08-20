from pathlib import Path
import os
import time

import pytest

from hermes_runtime.profile_manager import RunProfileManager, configured_context_length


@pytest.mark.parametrize("value", [8192, 32768, 131072])
def test_run_profile_renders_exact_context_and_header(monkeypatch, tmp_path: Path, value: int):
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", str(value))
    monkeypatch.setenv("HERMES_API_TOKEN", "gateway-secret")
    monkeypatch.setenv("LLM_API_URL", "http://provider.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("HERMES_PROFILE_UID", str(os.getuid()))
    monkeypatch.setenv("HERMES_PROFILE_GID", str(os.getgid()))
    manager = RunProfileManager(str(tmp_path))
    profile = manager.create(
        run_id=f"run-{value}", policy_profile="askpdf-deep-offline",
        context_token="signed.token", allowed_tools=["search_documents"],
        selected_model=f"model-{value}", selected_provider="lmstudio",
    )
    name = profile.name
    config = (tmp_path / name / "config.yaml").read_text()
    assert f"context_length: {value}" in config
    assert f'default: "model-{value}"' in config
    assert "base_url: \"http://provider.test/v1\"" in config
    assert "api_server:\n    enabled: false" in config
    assert "title_generation:\n    enabled: false" in config
    assert "X-AskPDF-Execution-Context" in config
    assert f"  {profile.mcp_server_name}:" in config
    assert f'name: "{profile.name}"' in config
    assert f'config_fingerprint: "{profile.activation_fingerprint}"' in config
    assert profile.mcp_server_name.startswith("askpdf_")
    assert "signed.token" not in config
    profile_env = (tmp_path / name / ".env").read_text()
    assert "API_SERVER_KEY=gateway-secret" in profile_env
    assert "LLM_API_URL=" not in profile_env
    assert "ASKPDF_MCP_EXECUTION_CONTEXT=signed.token" in profile_env
    assert "OPENAI_API_KEY=provider-secret" in profile_env
    assert all(value not in config for value in ("gateway-secret", "signed.token", "provider-secret"))
    assert ((tmp_path / name / ".env").stat().st_mode & 0o777) == 0o600
    assert manager.verify(profile) is True
    (tmp_path / name / "config.yaml").write_text(config + "\n# tampered")
    assert manager.verify(profile) is False
    manager.retire(profile)
    assert (tmp_path / name / "logs").is_dir()
    assert not (tmp_path / name / ".env").exists()
    retired_config = (tmp_path / name / "config.yaml").read_text()
    assert "mcp_servers: {}" in retired_config
    assert "title_generation:\n    enabled: false" in retired_config
    assert "X-AskPDF-Execution-Context" not in retired_config
    assert "provider.test" not in retired_config
    assert (tmp_path / name / ".askpdf-retired.json").is_file()


@pytest.mark.parametrize("value", ["", "true", "2047", "bad"])
def test_profile_context_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", value)
    with pytest.raises(RuntimeError):
        configured_context_length()


def test_run_profiles_are_isolated_and_stale_profiles_are_swept(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "24576")
    monkeypatch.setenv("HERMES_API_TOKEN", "gateway-secret")
    monkeypatch.setenv("LLM_API_URL", "http://provider.test/v1")
    monkeypatch.setenv("HERMES_PROFILE_UID", str(os.getuid()))
    monkeypatch.setenv("HERMES_PROFILE_GID", str(os.getgid()))
    manager = RunProfileManager(str(tmp_path))
    first = manager.create(run_id="run-one", policy_profile="askpdf-deep-offline", context_token="one.token", allowed_tools=["search_documents"], selected_model="model-one", selected_provider="lmstudio")
    second = manager.create(run_id="run-two", policy_profile="askpdf-deep-external", context_token="two.token", allowed_tools=["search_web"], selected_model="model-two", selected_provider="lmstudio")
    assert first.name != second.name
    assert "one.token" in (tmp_path / first.name / ".env").read_text()
    assert "two.token" in (tmp_path / second.name / ".env").read_text()
    manager.retire(first)
    old = time.time() - 3600
    os.utime(tmp_path / first.name, (old, old))
    assert manager.sweep_stale(max_age_seconds=60) == 1
    assert not (tmp_path / first.name).exists()
    assert (tmp_path / second.name).exists()


def test_generic_provider_preserves_pinned_hermes_64k_floor(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "32768")
    monkeypatch.setenv("HERMES_API_TOKEN", "gateway-secret")
    monkeypatch.setenv("LLM_API_URL", "http://provider.test/v1")
    manager = RunProfileManager(str(tmp_path))
    with pytest.raises(RuntimeError, match="at least 64000"):
        manager.create(
            run_id="run-generic", policy_profile="askpdf-deep-offline",
            context_token="signed.token", allowed_tools=["search_documents"],
            selected_model="generic-model", selected_provider="custom",
        )
