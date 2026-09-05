from __future__ import annotations

import os
from pathlib import Path

import pytest

from runtime_protocol.configuration import RuntimeConfigurationError, parse_bounded_ratio, validate_runtime_environment


def _environment() -> dict[str, str]:
    values = {
        "AGENT_RUNTIME_LEASE_SECONDS": "120",
        "AGENT_RUNTIME_CONNECT_TIMEOUT_SECONDS": "30",
        "AGENT_RUNTIME_WRITE_TIMEOUT_SECONDS": "300",
        "AGENT_RUNTIME_READ_TIMEOUT_SECONDS": "600",
        "AGENT_RUNTIME_RECONNECT_MAX_ATTEMPTS": "10",
        "AGENT_RUNTIME_RECONNECT_BACKOFF_SECONDS": "1",
        "AGENT_RUNTIME_RECONNECT_DEADLINE_SECONDS": "600",
        "AGENT_RUNTIME_OUTPUT_DELTA_FLUSH_SECONDS": "0.5",
        "AGENT_RUNTIME_OUTPUT_DELTA_FLUSH_BYTES": "8192",
        "AGENT_RUNTIME_SHUTDOWN_GRACE_SECONDS": "120",
        "AGENT_RUNTIME_CANCEL_CONFIRM_TIMEOUT_SECONDS": "120",
        "AGENT_RUNTIME_TERMINAL_CONFIRM_TIMEOUT_SECONDS": "120",
        "AGENT_EVENT_POLL_INTERVAL_SECONDS": "1",
        "AGENT_SSE_HEARTBEAT_INTERVAL_SECONDS": "12",
        "AGENT_CANCELLATION_POLL_INTERVAL_SECONDS": "0.5",
        "AGENT_RUNTIME_DEPENDENCY_REFRESH_SECONDS": "30",
        "AGENT_RUNTIME_DEPENDENCY_TIMEOUT_SECONDS": "60",
        "AGENT_RUNTIME_DEPENDENCY_STALE_SECONDS": "180",
        "AGENT_RUNTIME_DEPENDENCY_JITTER_RATIO": "0.1",
        "AGENT_RUNTIME_RECOVERY_INTERVAL_SECONDS": "30",
        "AGENT_RUNTIME_RECOVERY_BATCH_SIZE": "100",
        "AGENT_RUNTIME_RECOVERY_LOOP_ENABLED": "true",
        "MCP_OTEL_ENABLED": "false",
        "MCP_REQUEST_TIMEOUT_SECONDS": "600",
        "MCP_TRANSPORT": "loopback_http",
        "MCP_LOOPBACK_URL": "http://rag-service:8000/internal/mcp/",
        "NEXT_PUBLIC_AGENT_TASK_POLL_INTERVAL_MS": "2000",
        "NEXT_PUBLIC_AGENT_SSE_RECONNECT_INTERVAL_MS": "2000",
        "ASKPDF_AGENT_CHECKPOINTER": "postgres",
        "ASKPDF_AGENT_CHECKPOINTER_SETUP": "false",
        "AGENT_CHECKPOINT_DATABASE_URL": "postgresql://postgres:postgres@postgresql:5432/runtime_checkpoints",
        "AGENT_RUNTIME_EXECUTION_DATABASE_URL": "postgresql://postgres:postgres@postgresql:5432/runtime_checkpoints",
        "LANGGRAPH_RUNTIME_URL": "http://langgraph-runtime:8100",
        "LANGGRAPH_RUNTIME_TOKEN": "r" * 32,
        "LANGGRAPH_RUNTIME_BINDING_SECRET": "b" * 32,
        "LLM_AUTH_MODE": "none",
        "LLM_KEYLESS_PROVIDER": "local",
        "HERMES_MODEL_CONTEXT_LENGTH": "32768",
        "HERMES_MODEL_PROVIDER": "lmstudio",
        "HERMES_MCP_CONTEXT_SECRET": "x" * 32,
        "API_SERVER_KEY": "server-key",
        "HERMES_RUNTIME_URL": "http://hermes-runtime:8200",
        "DEFAULT_TOKEN_BUDGET": "8192",
        "REPLANS_LIMIT": "10",
        "MAX_CUSTOM_INSTRUCTIONS_CHARS": "2000",
        "MAX_SYSTEM_ROLE_CHARS": "500",
    }
    for suffix in (
        "MAX_MODEL_CALLS", "MAX_MODEL_TOKENS", "MAX_TOOL_CALLS", "MAX_ACTIVE_RUNTIME_MS",
        "MAX_DURATION_MS", "MAX_OUTPUT_CHARS", "MAX_EVENT_COUNT", "WAKE_LIMIT_SECONDS",
        "SUBAGENT_TIMEOUT_MS", "DISPATCH_TIMEOUT_MS", "WORKER_TIMEOUT_MS", "WEB_WORKER_TIMEOUT_MS",
    ):
        values[f"DEEP_AGENT_{suffix}"] = "100"
    for suffix in (
        "MAX_MODEL_CALLS", "MAX_MODEL_TOKENS", "MAX_TOOL_CALLS", "MAX_ACTIVE_RUNTIME_MS",
        "MAX_DURATION_MS", "MAX_OUTPUT_CHARS", "MAX_EVENT_COUNT", "WAKE_LIMIT_SECONDS",
        "SUBAGENT_TIMEOUT_MS", "DISPATCH_TIMEOUT_MS", "WORKER_TIMEOUT_MS", "WEB_WORKER_TIMEOUT_MS",
    ):
        values[f"DEEP_AGENT_LANGGRAPH_{suffix}"] = f"${{DEEP_AGENT_{suffix}}}"
    for suffix in (
        "MAX_MODEL_CALLS", "MAX_MODEL_TOKENS", "MAX_TOOL_CALLS", "MAX_ACTIVE_RUNTIME_MS",
        "MAX_DURATION_MS", "MAX_OUTPUT_CHARS", "MAX_EVENT_COUNT", "WAKE_LIMIT_SECONDS",
    ):
        values[f"DEEP_AGENT_HERMES_{suffix}"] = f"${{DEEP_AGENT_{suffix}}}"
    return values


def test_control_plane_ignores_framework_runtime_budget_variables():
    values = _environment()
    values.pop("DEEP_AGENT_LANGGRAPH_MAX_MODEL_CALLS")
    values.pop("DEEP_AGENT_HERMES_MAX_MODEL_CALLS")

    validated = validate_runtime_environment(service="control_plane", environ={**values, "COMPOSE_PROFILES": ""})

    assert "DEEP_AGENT_LANGGRAPH_MAX_MODEL_CALLS" not in validated.values


def test_control_plane_requires_langgraph_runtime_token():
    values = _environment()
    values.pop("LANGGRAPH_RUNTIME_TOKEN")
    with pytest.raises(RuntimeConfigurationError, match="LANGGRAPH_RUNTIME_TOKEN"):
        validate_runtime_environment(service="control_plane", environ={**values, "COMPOSE_PROFILES": ""})


def test_langgraph_limits_accept_non_default_values():
    values = _environment()
    values.update({
        "DEFAULT_TOKEN_BUDGET": "16384",
        "REPLANS_LIMIT": "20",
        "MAX_CUSTOM_INSTRUCTIONS_CHARS": "4000",
        "MAX_SYSTEM_ROLE_CHARS": "1000",
    })

    validated = validate_runtime_environment(service="langgraph", environ=values)

    assert {name: validated.get(name) for name in (
        "DEFAULT_TOKEN_BUDGET", "REPLANS_LIMIT",
        "MAX_CUSTOM_INSTRUCTIONS_CHARS", "MAX_SYSTEM_ROLE_CHARS",
    )} == {
        "DEFAULT_TOKEN_BUDGET": "16384",
        "REPLANS_LIMIT": "20",
        "MAX_CUSTOM_INSTRUCTIONS_CHARS": "4000",
        "MAX_SYSTEM_ROLE_CHARS": "1000",
    }


@pytest.mark.parametrize("name,value", [
    ("DEFAULT_TOKEN_BUDGET", ""),
    ("REPLANS_LIMIT", "0"),
    ("MAX_CUSTOM_INSTRUCTIONS_CHARS", "false"),
    ("MAX_SYSTEM_ROLE_CHARS", "not-an-integer"),
])
def test_langgraph_limits_reject_missing_or_invalid_values(name: str, value: str):
    values = _environment()
    values[name] = value

    with pytest.raises(RuntimeConfigurationError, match=name):
        validate_runtime_environment(service="langgraph", environ=values)


@pytest.mark.parametrize("value", ["0", "0.5"])
def test_jitter_ratio_accepts_inclusive_bounds(value):
    assert parse_bounded_ratio(value, name="jitter") == float(value)


@pytest.mark.parametrize("value", ["-0.01", "0.51", "nan", "inf", ""])
def test_jitter_ratio_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="finite ratio"):
        parse_bounded_ratio(value, name="jitter")


@pytest.mark.parametrize(
    "name,value",
    [
        ("DEEP_AGENT_MAX_TOOL_CALLS", "0"),
        ("DEEP_AGENT_LANGGRAPH_MAX_EVENT_COUNT", "not-an-integer"),
        ("AGENT_RUNTIME_RECONNECT_MAX_ATTEMPTS", "-1"),
        ("MCP_OTEL_ENABLED", "maybe"),
    ],
)
def test_invalid_runtime_configuration_is_rejected(name: str, value: str):
    values = _environment()
    values[name] = value

    with pytest.raises(RuntimeConfigurationError) as caught:
        validate_runtime_environment(service="langgraph", environ=values)

    assert name in str(caught.value)


def test_missing_values_are_aggregated_without_secret_values():
    values = _environment()
    values.pop("DEEP_AGENT_MAX_MODEL_CALLS")
    values.pop("AGENT_RUNTIME_READ_TIMEOUT_SECONDS")

    with pytest.raises(RuntimeConfigurationError) as caught:
        validate_runtime_environment(service="control_plane", environ={**values, "COMPOSE_PROFILES": ""})

    message = str(caught.value)
    assert "AGENT_RUNTIME_READ_TIMEOUT_SECONDS" in message
    assert "DEEP_AGENT_MAX_MODEL_CALLS" not in message


@pytest.mark.parametrize(
    "mutations",
    [
        {"DEEP_AGENT_MAX_TOOL_CALLS": "${DEEP_AGENT_MAX_MODEL_CALLS}", "DEEP_AGENT_MAX_MODEL_CALLS": "${DEEP_AGENT_MAX_TOOL_CALLS}"},
        {"DEEP_AGENT_MAX_TOOL_CALLS": "${DATABASE_URL}"},
        {"DEEP_AGENT_MAX_TOOL_CALLS": "${DEEP_AGENT_NOT_DEFINED}"},
    ],
)
def test_invalid_deep_agent_references_fail_startup(mutations):
    values = _environment()
    values.update(mutations)

    with pytest.raises(RuntimeConfigurationError) as caught:
        validate_runtime_environment(service="langgraph", environ=values)

    assert "DEEP_AGENT_MAX_TOOL_CALLS" in str(caught.value)


def test_langgraph_database_requirements_are_conditional():
    values = _environment()
    values.pop("AGENT_CHECKPOINT_DATABASE_URL")
    values.pop("AGENT_RUNTIME_EXECUTION_DATABASE_URL")

    validate_runtime_environment(service="control_plane", environ={**values, "COMPOSE_PROFILES": ""})

    with pytest.raises(RuntimeConfigurationError) as caught:
        validate_runtime_environment(service="langgraph", environ=values)
    assert "AGENT_RUNTIME_EXECUTION_DATABASE_URL" in str(caught.value)


def test_langgraph_checkpoint_database_never_falls_back_to_product_database():
    values = _environment()
    values.pop("AGENT_CHECKPOINT_DATABASE_URL")
    values["DATABASE_URL"] = "postgresql://product/database"

    with pytest.raises(RuntimeConfigurationError) as caught:
        validate_runtime_environment(service="langgraph", environ=values)
    assert "AGENT_CHECKPOINT_DATABASE_URL" in str(caught.value)


@pytest.mark.parametrize("service", ["langgraph", "hermes"])
def test_external_runtime_requires_loopback_mcp_transport(service):
    values = _environment()
    values["MCP_TRANSPORT"] = "in_process"
    with pytest.raises(RuntimeConfigurationError, match="MCP_TRANSPORT must be 'loopback_http'"):
        validate_runtime_environment(service=service, environ=values)


def test_control_plane_may_use_in_process_mcp_transport():
    values = _environment()
    values["MCP_TRANSPORT"] = "in_process"
    validate_runtime_environment(service="control_plane", environ=values)


@pytest.mark.parametrize(
    "mode,keyless_provider,api_key,expected",
    [
        ("required", "", "secret", True),
        ("required", "", "", False),
        ("none", "local", "", True),
        ("none", "", "", False),
        ("none", "remote", "", False),
    ],
)
def test_langgraph_model_authentication_is_explicit(mode, keyless_provider, api_key, expected):
    values = _environment()
    values.update({"LLM_AUTH_MODE": mode, "LLM_KEYLESS_PROVIDER": keyless_provider, "OPENAI_API_KEY": api_key})
    if expected:
        validate_runtime_environment(service="langgraph", environ=values)
    else:
        with pytest.raises(RuntimeConfigurationError):
            validate_runtime_environment(service="langgraph", environ=values)


def test_hermes_profile_bootstrap_does_not_require_http_runtime_settings():
    validated = validate_runtime_environment(
        service="hermes_profile",
        environ={
            "HERMES_MODEL_CONTEXT_LENGTH": "32768",
            "HERMES_MODEL_PROVIDER": "lmstudio",
            "HERMES_MCP_CONTEXT_SECRET": "x" * 32,
            "API_SERVER_KEY": "server-key",
            "HERMES_PROFILE_ROOT": "/opt/data/profiles",
            "HERMES_PROFILE_UID": "10000",
            "HERMES_PROFILE_GID": "10000",
        },
    )

    assert validated.get("HERMES_MODEL_PROVIDER") == "lmstudio"


def test_unused_environment_names_are_not_documented():
    example = Path(os.environ.get("ASKPDF_REPO_DIR", Path(__file__).parents[2])) / ".env.example"
    text = example.read_text()
    for name in (
        "AGENT_RUNTIME_MCP_READY_TIMEOUT_SECONDS",
        "AGENT_RUNTIME_PROVIDER_READY_TIMEOUT_SECONDS",
        "AGENT_RUNTIME_SCHEMA_AUTO_CREATE",
    ):
        assert name not in text
