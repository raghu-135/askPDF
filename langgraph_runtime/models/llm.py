"""Execution-scoped OpenAI-compatible model client for LangGraph nodes."""

from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_openai import ChatOpenAI


def _positive_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    result = int(value)
    if result < 1:
        raise RuntimeError(f"{name} must be positive")
    return result


DEFAULT_TOKEN_BUDGET = _positive_int("DEFAULT_TOKEN_BUDGET", 8192)
REPLANS_LIMIT = _positive_int("REPLANS_LIMIT", 10)
MAX_CUSTOM_INSTRUCTIONS_CHARS = _positive_int("MAX_CUSTOM_INSTRUCTIONS_CHARS", 2000)
MAX_SYSTEM_ROLE_CHARS = _positive_int("MAX_SYSTEM_ROLE_CHARS", 500)


def provider_configuration(base_url_override: str | None = None) -> tuple[str, dict[str, str], str]:
    """Return the validated provider URL, safe request headers, and API key."""
    base_url = (base_url_override if base_url_override is not None else os.getenv("LLM_API_URL", "")).strip()
    if not base_url:
        raise RuntimeError("LLM_API_URL is required by langgraph-runtime")
    auth_mode = os.getenv("LLM_AUTH_MODE", "").strip().lower()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if auth_mode == "required":
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_AUTH_MODE=required")
        return base_url, {"authorization": f"Bearer {api_key}"}, api_key
    if auth_mode == "none":
        provider = os.getenv("LLM_KEYLESS_PROVIDER", "").strip().lower()
        if provider not in {"lmstudio", "ollama", "local"}:
            raise RuntimeError("LLM_KEYLESS_PROVIDER must identify an allowed local provider")
        return base_url, {}, ""
    raise RuntimeError("LLM_AUTH_MODE must be 'required' or 'none'")


def get_llm(model_name: str, temperature: float = 0.0, *, own_async_transport: bool = True) -> ChatOpenAI:
    base_url, headers, api_key = provider_configuration()
    client = httpx.AsyncClient() if own_async_transport else None
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
        # The OpenAI SDK rejects an empty key even for local/keyless
        # OpenAI-compatible servers. This placeholder is not a credential;
        # keyless readiness probes still send no Authorization header.
        api_key=api_key or "not-needed",
        default_headers=headers or None,
        http_async_client=client,
    )


async def close_model_client(model: Any) -> None:
    client = getattr(model, "http_async_client", None)
    close = getattr(client, "aclose", None) if client is not None else None
    if close is not None:
        await close()
