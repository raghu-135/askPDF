from __future__ import annotations

from pathlib import Path

import pytest

from runtime_protocol.configuration import RuntimeConfigurationError, validate_runtime_environment


def _environment() -> dict[str, str]:
    values = {
        "AGENT_RUNTIME_MODE": "external",
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
        "MCP_TRANSPORT": "in_process",
        "NEXT_PUBLIC_AGENT_TASK_POLL_INTERVAL_MS": "2000",
        "NEXT_PUBLIC_AGENT_SSE_RECONNECT_INTERVAL_MS": "2000",
        "ASKPDF_AGENT_CHECKPOINTER": "postgres",
        "AGENT_CHECKPOINT_DATABASE_URL": "postgresql://postgres:postgres@postgresql:5432/runtime_checkpoints",
        "AGENT_RUNTIME_EXECUTION_DATABASE_URL": "postgresql://postgres:postgres@postgresql:5432/runtime_checkpoints",
        "LANGGRAPH_RUNTIME_URL": "http://langgraph-runtime:8100",
        "HERMES_MODEL_CONTEXT_LENGTH": "32768",
        "HERMES_MODEL_PROVIDER": "lmstudio",
        "HERMES_MCP_CONTEXT_SECRET": "x" * 32,
        "API_SERVER_KEY": "server-key",
        "HERMES_RUNTIME_URL": "http://hermes-runtime:8200",
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


def test_framework_budget_aliases_resolve_and_explicit_override_wins():
    values = _environment()
    values["DEEP_AGENT_HERMES_MAX_ACTIVE_RUNTIME_MS"] = "250"

    validated = validate_runtime_environment(service="control_plane", environ={**values, "COMPOSE_PROFILES": ""})

    assert validated.get("DEEP_AGENT_HERMES_MAX_ACTIVE_RUNTIME_MS") == "250"


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
        validate_runtime_environment(service="control_plane", environ={**values, "COMPOSE_PROFILES": ""})

    assert name in str(caught.value)


def test_missing_values_are_aggregated_without_secret_values():
    values = _environment()
    values.pop("DEEP_AGENT_MAX_MODEL_CALLS")
    values.pop("AGENT_RUNTIME_READ_TIMEOUT_SECONDS")
    values["HERMES_MCP_CONTEXT_SECRET"] = "secret-value"

    with pytest.raises(RuntimeConfigurationError) as caught:
        validate_runtime_environment(service="control_plane", environ={**values, "COMPOSE_PROFILES": "hermes"})

    message = str(caught.value)
    assert "DEEP_AGENT_LANGGRAPH_MAX_MODEL_CALLS" in message
    assert "AGENT_RUNTIME_READ_TIMEOUT_SECONDS" in message
    assert "HERMES_MCP_CONTEXT_SECRET must contain at least 32 characters" in message
    assert "secret-value" not in message


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
        validate_runtime_environment(service="control_plane", environ={**values, "COMPOSE_PROFILES": ""})

    assert "DEEP_AGENT_MAX_TOOL_CALLS" in str(caught.value)


def test_langgraph_database_requirements_are_conditional():
    values = _environment()
    values.pop("AGENT_CHECKPOINT_DATABASE_URL")
    values.pop("AGENT_RUNTIME_EXECUTION_DATABASE_URL")
    values.pop("LANGGRAPH_RUNTIME_URL")

    validate_runtime_environment(service="control_plane", environ={**values, "COMPOSE_PROFILES": ""})

    with pytest.raises(RuntimeConfigurationError) as caught:
        validate_runtime_environment(service="langgraph", environ=values)
    assert "AGENT_RUNTIME_EXECUTION_DATABASE_URL" in str(caught.value)


def test_unused_environment_names_are_not_documented():
    example = Path(__file__).parents[2] / ".env.example"
    text = example.read_text()
    for name in (
        "AGENT_RUNTIME_MCP_READY_TIMEOUT_SECONDS",
        "AGENT_RUNTIME_PROVIDER_READY_TIMEOUT_SECONDS",
        "AGENT_RUNTIME_SCHEMA_AUTO_CREATE",
    ):
        assert name not in text
