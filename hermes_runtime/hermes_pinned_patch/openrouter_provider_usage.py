"""OpenRouter provider-reported usage for askPDF's pinned Hermes gateway.

Hermes terminal run ``usage`` payloads include token counters only. askPDF's
Debug Trace needs the provider-reported USD ``cost`` when OpenRouter returns it
in ``usage`` — never token-derived estimates.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping
from urllib.parse import urlparse

from runtime_protocol.provider_usage import provider_cost_usd

_ATTR = "_askpdf_provider_cost_usd"
logger = logging.getLogger("askpdf.hermes_pinned_patch.openrouter")


def _usage_mapping(usage: Any) -> Mapping[str, Any] | None:
    if usage is None:
        return None
    if isinstance(usage, Mapping):
        return usage
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except TypeError:
            dumped = model_dump(warnings=False)
        if isinstance(dumped, Mapping):
            return dumped
    extracted: dict[str, Any] = {}
    for key in ("cost", "provider_cost_usd", "total_cost", "prompt_tokens", "completion_tokens", "total_tokens"):
        if hasattr(usage, key):
            value = getattr(usage, key)
            if value is not None:
                extracted[key] = value
    return extracted or None


def openrouter_usage_accounting_enabled(*, base_url: str, model: str, provider: str | None = None) -> bool:
    """True when chat completions should ask OpenRouter for usage accounting."""

    accounting_flag = os.environ.get("ASKPDF_OPENROUTER_USAGE_ACCOUNTING", "").strip().lower()
    if accounting_flag in {"0", "false", "no", "off"}:
        return False
    if accounting_flag in {"1", "true", "yes", "on"}:
        return True
    if (provider or "").strip().lower() == "openrouter":
        return True
    if "openrouter.ai" in str(base_url or "").casefold():
        return True
    parsed = urlparse(str(base_url or "").strip())
    host = (parsed.hostname or "").casefold()
    if host in {"host.docker.internal", "127.0.0.1", "localhost"} and "/" in str(model or ""):
        proxy_port = int(os.environ.get("OPENROUTER_PROXY_PORT", "12345") or "12345")
        if parsed.port in {None, proxy_port}:
            return True
    return False


def merge_openrouter_usage_request(api_kwargs: dict[str, Any], *, base_url: str, model: str, provider: str | None) -> dict[str, Any]:
    """Merge OpenRouter usage-accounting fields into chat-completion kwargs."""

    if not openrouter_usage_accounting_enabled(base_url=base_url, model=model, provider=provider):
        return api_kwargs
    merged = dict(api_kwargs)
    # The OpenAI Python client only accepts documented parameters; OpenRouter's
    # ``usage`` extension must ride in ``extra_body``.
    extra_body = merged.get("extra_body")
    if not isinstance(extra_body, Mapping):
        extra_body = {}
    usage = extra_body.get("usage")
    if not isinstance(usage, Mapping):
        usage = {}
    if usage.get("include") is not True:
        extra_body = {**dict(extra_body), "usage": {**dict(usage), "include": True}}
        merged["extra_body"] = extra_body
        logger.debug("OpenRouter usage accounting: enabled usage.include for model %s", model)
    merged.pop("usage", None)
    stream_options = merged.get("stream_options")
    if isinstance(stream_options, Mapping):
        if stream_options.get("include_usage") is not True:
            merged["stream_options"] = {**dict(stream_options), "include_usage": True}
    elif merged.get("stream") and stream_options is None:
        merged["stream_options"] = {"include_usage": True}
    return merged


def provider_cost_from_response_usage(usage: Any) -> float | None:
    return provider_cost_usd(_usage_mapping(usage))


def accumulate_agent_provider_cost(agent: Any, response: Any) -> None:
    """Fold provider-reported ``usage.cost`` into a session counter on the agent."""

    usage = getattr(response, "usage", None)
    cost = provider_cost_from_response_usage(usage)
    if cost is None:
        return
    total = float(getattr(agent, _ATTR, 0) or 0) + cost
    setattr(agent, _ATTR, round(total, 8))
    logger.debug(
        "OpenRouter provider usage cost %.8f USD (session total %.8f USD)",
        cost,
        getattr(agent, _ATTR),
    )


def agent_provider_cost_usd(agent: Any) -> float | None:
    raw = getattr(agent, _ATTR, None)
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return value


def enrich_terminal_usage(usage: dict[str, Any], agent: Any) -> dict[str, Any]:
    """Add provider-reported cost fields to Hermes' token-only terminal usage."""

    cost = agent_provider_cost_usd(agent)
    if cost is None:
        return usage
    enriched = dict(usage)
    enriched["cost"] = cost
    enriched["provider_cost_usd"] = cost
    return enriched
