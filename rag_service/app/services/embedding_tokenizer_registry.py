"""Built-in tokenizer metadata for remote embedding models.

The LLM provider only returns vectors. Chunk packing still needs the matching
Hugging Face tokenizer, input limit, and optional query/document prefixes.
EMBEDDING_TOKENIZER_CONFIG_JSON overrides or extends these defaults.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping


# Qwen3-Embedding query formatting from the model card.
_QWEN3_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)
_BERT_LIMIT = 480
_QWEN_LIMIT = 8192
_BGE_M3_LIMIT = 8192

_QWEN3 = {
    "effective_input_limit": _QWEN_LIMIT,
    "query_prefix": _QWEN3_QUERY_PREFIX,
}
_E5 = {
    "effective_input_limit": _BERT_LIMIT,
    "prefix": "passage: ",
    "query_prefix": "query: ",
}
_BGE_EN = {
    "effective_input_limit": _BERT_LIMIT,
    "query_prefix": "Represent this sentence for searching relevant passages: ",
}
_MINILM = {"effective_input_limit": _BERT_LIMIT}
_GTE = {"effective_input_limit": _BERT_LIMIT}


def _entry(tokenizer: str, **fields: Any) -> dict[str, Any]:
    return {"tokenizer": tokenizer, **fields}


# Keys are provider model ids (OpenRouter, LM Studio, Ollama aliases).
BUILTIN_EMBEDDING_TOKENIZERS: dict[str, dict[str, Any]] = {
    "qwen/qwen3-embedding-8b": _entry("Qwen/Qwen3-Embedding-8B", **_QWEN3),
    "qwen/qwen3-embedding-4b": _entry("Qwen/Qwen3-Embedding-4B", **_QWEN3),
    "thenlper/gte-base": _entry("thenlper/gte-base", **_GTE),
    "thenlper/gte-large": _entry("thenlper/gte-large", **_GTE),
    "intfloat/e5-base-v2": _entry("intfloat/e5-base-v2", **_E5),
    "intfloat/e5-large-v2": _entry("intfloat/e5-large-v2", **_E5),
    "intfloat/multilingual-e5-large": _entry("intfloat/multilingual-e5-large", **_E5),
    "baai/bge-m3": _entry("BAAI/bge-m3", effective_input_limit=_BGE_M3_LIMIT),
    "baai/bge-base-en-v1.5": _entry("BAAI/bge-base-en-v1.5", **_BGE_EN),
    "baai/bge-large-en-v1.5": _entry("BAAI/bge-large-en-v1.5", **_BGE_EN),
    "sentence-transformers/all-minilm-l6-v2": _entry(
        "sentence-transformers/all-MiniLM-L6-v2", **_MINILM
    ),
    "sentence-transformers/all-minilm-l12-v2": _entry(
        "sentence-transformers/all-MiniLM-L12-v2", **_MINILM
    ),
    "sentence-transformers/all-mpnet-base-v2": _entry(
        "sentence-transformers/all-mpnet-base-v2", **_MINILM
    ),
    "sentence-transformers/paraphrase-minilm-l6-v2": _entry(
        "sentence-transformers/paraphrase-MiniLM-L6-v2", **_MINILM
    ),
    "sentence-transformers/multi-qa-mpnet-base-dot-v1": _entry(
        "sentence-transformers/multi-qa-mpnet-base-dot-v1", **_MINILM
    ),
    "liquid/lfm-2.5-embedding-350m": _entry(
        "LiquidAI/LFM2.5-Embedding-350M",
        effective_input_limit=_BERT_LIMIT,
    ),
    "liquid/lfm-2.5-embedding-350m:free": _entry(
        "LiquidAI/LFM2.5-Embedding-350M",
        effective_input_limit=_BERT_LIMIT,
    ),
    "nomic-ai/nomic-embed-text-v1.5": _entry(
        "nomic-ai/nomic-embed-text-v1.5",
        effective_input_limit=_BERT_LIMIT,
        prefix="search_document: ",
        query_prefix="search_query: ",
    ),
    "text-embedding-nomic-embed-text-v1.5": _entry(
        "nomic-ai/nomic-embed-text-v1.5",
        effective_input_limit=_BERT_LIMIT,
        prefix="search_document: ",
        query_prefix="search_query: ",
    ),
    "nomic-embed-text": _entry(
        "nomic-ai/nomic-embed-text-v1.5",
        effective_input_limit=_BERT_LIMIT,
        prefix="search_document: ",
        query_prefix="search_query: ",
    ),
}


def _strip_provider_variant(model: str) -> str:
    name = str(model or "").strip()
    if name.endswith(":free"):
        return name[: -len(":free")]
    return name


def _external_mapping() -> dict[str, Any]:
    raw = os.environ.get("EMBEDDING_TOKENIZER_CONFIG_JSON", "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("EMBEDDING_TOKENIZER_CONFIG_JSON must be valid JSON") from exc
    if not isinstance(value, dict):
        return {}
    return {
        str(key).strip(): dict(fields)
        for key, fields in value.items()
        if str(key).strip() and isinstance(fields, Mapping)
    }


def lookup_embedding_tokenizer_settings(model: str) -> dict[str, Any] | None:
    """Return tokenizer settings without loading Hugging Face files."""
    name = str(model or "").strip()
    if not name:
        return None
    overlay = _external_mapping()
    if name in overlay:
        return dict(overlay[name])
    builtin = BUILTIN_EMBEDDING_TOKENIZERS.get(name)
    if builtin is None:
        builtin = BUILTIN_EMBEDDING_TOKENIZERS.get(_strip_provider_variant(name))
    if builtin is None:
        return None
    return dict(builtin)


def is_registered_embedding_model(model: str) -> bool:
    return lookup_embedding_tokenizer_settings(model) is not None


def unique_tokenizer_identities() -> list[tuple[str, str | None]]:
    """Hugging Face tokenizer ids to prefetch; revision None means default."""
    seen: dict[str, str | None] = {}
    for settings in BUILTIN_EMBEDDING_TOKENIZERS.values():
        identity = str(settings.get("tokenizer") or "").strip()
        if not identity or identity in seen:
            continue
        revision = str(settings.get("revision") or "").strip() or None
        seen[identity] = revision
    return sorted(seen.items(), key=lambda item: item[0].casefold())


__all__ = [
    "BUILTIN_EMBEDDING_TOKENIZERS",
    "is_registered_embedding_model",
    "lookup_embedding_tokenizer_settings",
    "unique_tokenizer_identities",
]
