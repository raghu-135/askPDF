"""Backward-compatible imports for the JSON embedding model catalog."""

from __future__ import annotations

from typing import Any

from app.services.embedding_model_catalog import (
    has_exact_hf_config,
    load_catalog,
    lookup_provider_model,
    resolve_settings_for_provider,
    unique_hf_tokenizer_identities,
)


def lookup_embedding_tokenizer_settings(model: str) -> dict[str, Any] | None:
    """Return tokenizer settings for exact-tier catalog entries only (legacy helper)."""
    resolved = resolve_settings_for_provider(model)
    if resolved.tier != "exact" or not resolved.tokenizer:
        return None
    return {
        "tokenizer": resolved.tokenizer,
        "revision": resolved.revision,
        "effective_input_limit": resolved.effective_input_limit,
        "prefix": resolved.prefix,
        "suffix": resolved.suffix,
        **({"query_prefix": resolved.query_prefix} if resolved.query_prefix is not None else {}),
        **({"query_suffix": resolved.query_suffix} if resolved.query_suffix is not None else {}),
    }


def is_registered_embedding_model(model: str) -> bool:
    """True when the provider id has an exact Hugging Face catalog entry."""
    return has_exact_hf_config(model)


__all__ = [
    "is_registered_embedding_model",
    "load_catalog",
    "lookup_embedding_tokenizer_settings",
    "lookup_provider_model",
    "unique_hf_tokenizer_identities",
]