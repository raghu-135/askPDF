"""Execution-scoped OpenAI-compatible model client for LangGraph nodes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_openai import ChatOpenAI
from runtime_protocol.configuration import LANGGRAPH_LIMIT_NAMES, parse_required_positive_int


@dataclass(frozen=True)
class LangGraphLimits:
    default_token_budget: int
    replans_limit: int
    max_custom_instructions_chars: int
    max_system_role_chars: int


def load_runtime_limits(environ: dict[str, str] | None = None) -> LangGraphLimits:
    values = os.environ if environ is None else environ
    parsed = {
        name: parse_required_positive_int(name, values.get(name))
        for name in LANGGRAPH_LIMIT_NAMES
    }
    return LangGraphLimits(
        default_token_budget=parsed["DEFAULT_TOKEN_BUDGET"],
        replans_limit=parsed["REPLANS_LIMIT"],
        max_custom_instructions_chars=parsed["MAX_CUSTOM_INSTRUCTIONS_CHARS"],
        max_system_role_chars=parsed["MAX_SYSTEM_ROLE_CHARS"],
    )


def configure_runtime_limits(environ: dict[str, str] | None = None) -> LangGraphLimits:
    """Validate and return limits for an explicit process/bootstrap boundary."""
    return load_runtime_limits(environ)


def runtime_limits(environ: dict[str, str] | None = None) -> LangGraphLimits:
    """Load immutable limits for a runtime operation; never retain global state."""
    return load_runtime_limits(environ)


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
