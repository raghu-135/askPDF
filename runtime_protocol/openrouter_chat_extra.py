"""OpenRouter chat-completion extensions shared by control plane and runtimes."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

def openrouter_chat_extra_body(*, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Return extra_body kwargs for ChatOpenAI when calling OpenRouter.

    ``ASKPDF_OPENROUTER_CHAT_EXTRA_BODY_JSON`` overrides everything. When unset,
    ``ASKPDF_LLM_REASONING_EFFORT`` sets OpenRouter ``reasoning.effort`` when
    non-empty. Unset or ``default`` omits reasoning extensions (normal app default).
    """
    values = os.environ if environ is None else environ
    raw = (values.get("ASKPDF_OPENROUTER_CHAT_EXTRA_BODY_JSON") or "").strip()
    if raw:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("ASKPDF_OPENROUTER_CHAT_EXTRA_BODY_JSON must be a JSON object")
        return parsed
    effort = (values.get("ASKPDF_LLM_REASONING_EFFORT") or "").strip()
    if not effort:
        return {}
    if effort.lower() == "default":
        return {}
    return {"reasoning": {"effort": effort}}


def merge_chat_openai_kwargs(api_kwargs: dict[str, Any], *, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    extra = openrouter_chat_extra_body(environ=environ)
    if not extra:
        return api_kwargs
    merged = dict(api_kwargs)
    body = merged.get("extra_body")
    if not isinstance(body, Mapping):
        body = {}
    merged["extra_body"] = {**dict(body), **extra}
    return merged


def merge_chat_completions_payload(payload: dict[str, Any], *, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Merge OpenRouter extensions into a raw /chat/completions JSON body."""
    extra = openrouter_chat_extra_body(environ=environ)
    if not extra:
        return payload
    return {**payload, **extra}