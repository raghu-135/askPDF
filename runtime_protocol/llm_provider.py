"""Shared OpenAI-compatible LLM server configuration.

Control plane, LangGraph, and Hermes all talk to one base URL. Auth is a single
optional API key: a nonempty OPENAI_API_KEY sends Authorization: Bearer; an
empty key sends no Authorization header.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

LLM_SDK_PLACEHOLDER_API_KEY = "not-needed"


class LlmProviderConfigurationError(RuntimeError):
    """Raised when the shared LLM server URL cannot be used."""


@dataclass(frozen=True)
class LlmProviderConfiguration:
    base_url: str
    request_headers: dict[str, str]
    credential: str
    sdk_api_key: str

    def openai_sdk_default_headers(self) -> dict[str, str] | None:
        """Headers for ChatOpenAI/OpenAIEmbeddings. Authorization is api_key."""
        return openai_sdk_default_headers(self.request_headers)


def normalize_llm_api_url(url: str) -> str:
    """Return an OpenAI-compatible base URL that ends with /v1."""
    value = (url or "").strip().rstrip("/")
    if not value:
        raise LlmProviderConfigurationError("LLM_API_URL is required")
    if value.endswith("/v1"):
        return value
    return f"{value}/v1"


def openai_sdk_default_headers(headers: Mapping[str, str]) -> dict[str, str] | None:
    """Drop Authorization so the SDK does not send a duplicate Bearer token."""
    filtered = {
        key: value
        for key, value in headers.items()
        if key.lower() != "authorization"
    }
    return filtered or None


def llm_provider_configuration(
    *,
    base_url: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> LlmProviderConfiguration:
    """Resolve the shared LLM server URL and optional Bearer credential."""
    values = os.environ if environ is None else environ
    raw = values.get("LLM_API_URL", "") if base_url is None else base_url
    normalized = normalize_llm_api_url(raw if raw is not None else "")
    credential = (values.get("OPENAI_API_KEY") or "").strip()
    if credential:
        return LlmProviderConfiguration(
            base_url=normalized,
            request_headers={"Authorization": f"Bearer {credential}"},
            credential=credential,
            sdk_api_key=credential,
        )
    return LlmProviderConfiguration(
        base_url=normalized,
        request_headers={},
        credential="",
        sdk_api_key=LLM_SDK_PLACEHOLDER_API_KEY,
    )
