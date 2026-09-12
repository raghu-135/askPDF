"""Execution-scoped OpenAI-compatible model client for LangGraph nodes."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx
from langchain_openai import ChatOpenAI
from runtime_protocol.configuration import LANGGRAPH_LIMIT_NAMES, parse_required_positive_int


@dataclass(frozen=True)
class LangGraphLimits:
    default_token_budget: int
    replans_limit: int
    max_custom_instructions_chars: int
    max_system_role_chars: int


_execution_client: ContextVar[httpx.AsyncClient | None] = ContextVar(
    "langgraph_execution_client", default=None
)


@asynccontextmanager
async def model_client_scope() -> AsyncIterator[httpx.AsyncClient]:
    """Own exactly one provider client for the lifetime of one execution."""
    client = httpx.AsyncClient()
    token = _execution_client.set(client)
    try:
        yield client
    finally:
        _execution_client.reset(token)
        await client.aclose()


def execution_model_client(config: Any = None) -> httpx.AsyncClient:
    """Return the execution-owned client, failing when called out of scope."""
    configured = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    client = configured.get("model_client") or _execution_client.get()
    if not isinstance(client, httpx.AsyncClient):
        raise RuntimeError("LangGraph model client is unavailable outside an execution scope")
    return client


def current_execution_model_client() -> httpx.AsyncClient | None:
    return _execution_client.get()


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


def get_llm(
    model_name: str,
    temperature: float = 0.0,
    *,
    http_async_client: httpx.AsyncClient | None = None,
) -> ChatOpenAI:
    base_url, headers, api_key = provider_configuration()
    client = http_async_client or _execution_client.get()
    if client is None:
        raise RuntimeError("LangGraph model client is unavailable outside an execution scope")
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
