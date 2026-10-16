"""Execution-scoped OpenAI-compatible model client for LangGraph nodes."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx
from langchain_openai import ChatOpenAI
from openai import BaseModel as OpenAIBaseModel
from runtime_protocol.configuration import LANGGRAPH_LIMIT_NAMES, parse_required_positive_int
from runtime_protocol.llm_provider import llm_provider_configuration, openai_sdk_default_headers

_REASONING_RESPONSE_FIELDS = (
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "reasoning_summary",
    "reasoning_text",
    "thinking",
    "thoughts",
)


class ReasoningChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant that preserves OpenAI-compatible reasoning extensions."""

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}}
            )
            if isinstance(response, OpenAIBaseModel)
            else {}
        )

        choices = response_dict.get("choices", []) if isinstance(response_dict, dict) else []
        for generation, choice in zip(result.generations, choices):
            message_dict = choice.get("message", {}) if isinstance(choice, dict) else {}
            if not isinstance(message_dict, dict):
                continue
            preserved = {
                key: value
                for key, value in message_dict.items()
                if key in _REASONING_RESPONSE_FIELDS and value
            }
            if preserved:
                generation.message.additional_kwargs.update(preserved)

        return result


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
    """Return the shared LLM server URL, request headers, and SDK API key."""
    provider = llm_provider_configuration(base_url=base_url_override)
    return provider.base_url, dict(provider.request_headers), provider.sdk_api_key


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
    return ReasoningChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        default_headers=openai_sdk_default_headers(headers),
        http_async_client=client,
    )
