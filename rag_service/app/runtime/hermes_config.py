"""Authoritative deployment configuration for the Hermes runtime."""

from __future__ import annotations

import os
from runtime_protocol.hermes_contract import HERMES_MIN_CONTEXT_LENGTH


class HermesConfigurationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def hermes_runtime_enabled() -> bool:
    profiles = {
        value.strip().lower()
        for value in os.getenv("COMPOSE_PROFILES", "").split(",")
        if value.strip()
    }
    return "hermes" in profiles


def hermes_model_context_length(*, required: bool = True) -> int | None:
    """Return the deployment-owned Hermes context window without defaults."""

    raw = os.getenv("HERMES_MODEL_CONTEXT_LENGTH", "").strip()
    if not raw:
        if required:
            raise HermesConfigurationError(
                "hermes_context_length_unconfigured",
                "Hermes model context length is required when Hermes is enabled",
            )
        return None
    if raw.lower() in {"true", "false", "yes", "no", "on", "off"}:
        raise HermesConfigurationError(
            "hermes_context_length_invalid",
            "Hermes model context length must be an integer",
        )
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise HermesConfigurationError(
            "hermes_context_length_invalid",
            "Hermes model context length must be an integer",
        ) from exc
    if value < HERMES_MIN_CONTEXT_LENGTH:
        raise HermesConfigurationError(
            "hermes_context_length_invalid",
            f"Hermes model context length must be at least {HERMES_MIN_CONTEXT_LENGTH}",
        )
    return value


def validate_hermes_model_compatibility() -> int:
    """Validate the operator-owned Hermes context window."""

    context_length = hermes_model_context_length()
    assert context_length is not None
    return context_length
