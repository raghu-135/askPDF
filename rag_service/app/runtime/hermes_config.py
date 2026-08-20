"""Authoritative deployment configuration for the Hermes runtime."""

from __future__ import annotations

import os


HERMES_MIN_CONTEXT_LENGTH = 2048
HERMES_PINNED_MIN_CONTEXT_LENGTH = 64_000


class HermesConfigurationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def hermes_runtime_enabled() -> bool:
    return os.getenv("HERMES_RUNTIME_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


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


def hermes_model_provider() -> str:
    """Return the Hermes provider selected at the deployment boundary."""

    provider = os.getenv("HERMES_MODEL_PROVIDER", "custom").strip().lower()
    if not provider or any(character.isspace() for character in provider):
        raise HermesConfigurationError(
            "hermes_model_provider_invalid",
            "Hermes model provider must be a non-empty provider identifier",
        )
    return provider


def validate_hermes_model_compatibility() -> tuple[int, str]:
    """Validate constraints imposed by the pinned Hermes revision."""

    context_length = hermes_model_context_length()
    assert context_length is not None
    provider = hermes_model_provider()
    # bdd0a79 permits an explicitly configured smaller window only for its
    # first-class LM Studio provider. All other providers enforce the 64K floor.
    if provider != "lmstudio" and context_length < HERMES_PINNED_MIN_CONTEXT_LENGTH:
        raise HermesConfigurationError(
            "hermes_context_length_provider_incompatible",
            f"Hermes provider {provider!r} requires a context length of at least "
            f"{HERMES_PINNED_MIN_CONTEXT_LENGTH}; LM Studio permits an explicitly configured smaller value",
        )
    return context_length, provider
