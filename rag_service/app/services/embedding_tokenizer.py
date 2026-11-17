"""Embedding tokenizer configuration and exact input-budget accounting."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

from app.services.document_pipeline import TokenCounter
from app.services.embedding_model_catalog import resolve_settings_for_provider


class EmbeddingTokenizerUnavailableError(RuntimeError):
    """Raised when the exact tokenizer required by an embedding model is unavailable."""


def _split_oversized_fragment(text: str, limit: int, count) -> list[str]:
    """Split an unbroken token at tokenizer-safe character boundaries."""
    fragment = str(text)
    if not fragment:
        return []
    if limit <= 0:
        raise ValueError("token limit must be positive")
    if count(fragment) <= limit:
        return [fragment]
    pieces: list[str] = []
    start = 0
    while start < len(fragment):
        low, high = start + 1, len(fragment)
        best = start
        while low <= high:
            middle = (low + high) // 2
            if count(fragment[start:middle]) <= limit:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best == start:
            raise ValueError("tokenizer cannot represent a single character within the input budget")
        pieces.append(fragment[start:best])
        start = best
    return pieces


def _pack_limit(limit: int, margin: float | None) -> int:
    if margin is None or margin <= 0 or margin > 1:
        return limit
    return max(1, int(limit * margin))


def _build_token_counter(
    config: "EmbeddingTokenizerConfig",
    count_fn,
) -> TokenCounter:
    def count(text: str) -> int:
        formatted = config.format_document_input(text)
        return count_fn(formatted)

    def split(text: str, limit: int) -> Sequence[str]:
        words = text.split()
        output: list[str] = []
        current: list[str] = []
        for word in words:
            if count(word) > limit:
                if current:
                    output.append(" ".join(current))
                    current = []
                output.extend(_split_oversized_fragment(word, limit, count))
                continue
            candidate = " ".join([*current, word])
            if current and count(candidate) > limit:
                output.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            output.append(" ".join(current))
        return output

    return TokenCounter(count=count, split=split)


def _tiktoken_counter(encoding_name: str):
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)

    def count_ids(formatted: str) -> int:
        return len(encoding.encode(formatted, disallowed_special=()))

    return count_ids


@dataclass(frozen=True)
class EmbeddingTokenizerConfig:
    identity: str
    revision: str | None
    effective_input_limit: int
    tier: str = "exact"
    required_prefix: str = ""
    required_suffix: str = ""
    query_prefix: str | None = None
    query_suffix: str | None = None

    def format_document_input(self, text: str) -> str:
        return f"{self.required_prefix}{text}{self.required_suffix}"

    def format_query_input(self, text: str) -> str:
        prefix = self.required_prefix if self.query_prefix is None else self.query_prefix
        suffix = self.required_suffix if self.query_suffix is None else self.query_suffix
        return f"{prefix}{text}{suffix}"

    @property
    def fingerprint(self) -> str:
        return (
            f"{self.tier}:{self.identity}@{self.revision or 'default'}:{self.effective_input_limit}:"
            f"{self.required_prefix}:{self.required_suffix}:"
            f"{self.query_prefix}:{self.query_suffix}"
        )


def _local_embedding_settings(model: str) -> dict[str, Any]:
    return {
        "tier": "local",
        "tokenizer": os.environ.get("LOCAL_EMBEDDING_TOKENIZER", model),
        "revision": os.environ.get("LOCAL_EMBEDDING_TOKENIZER_REVISION") or None,
        "effective_input_limit": int(os.environ.get("LOCAL_EMBEDDING_INPUT_LIMIT", "512") or "512"),
        "prefix": os.environ.get("LOCAL_EMBEDDING_FORMAT_PREFIX", ""),
        "suffix": os.environ.get("LOCAL_EMBEDDING_FORMAT_SUFFIX", ""),
    }


@lru_cache(maxsize=64)
def resolve_embedding_tokenizer(model: str) -> tuple[EmbeddingTokenizerConfig, TokenCounter]:
    """Resolve local, exact HF, or generic tiktoken token accounting for a model id."""
    model = str(model or "").strip()
    if not model:
        raise ValueError("embedding model is required")

    local_model = os.environ.get("LOCAL_EMBEDDING_MODEL", "").strip()
    try:
        if model == local_model:
            settings = _local_embedding_settings(model)
            tier = "local"
            identity = str(settings["tokenizer"])
            limit = int(settings["effective_input_limit"])
            revision = settings["revision"]
            query_prefix = None
            query_suffix = None
        else:
            resolved = resolve_settings_for_provider(model)
            tier = resolved.tier
            identity = resolved.tokenizer or f"generic:{resolved.encoding or 'cl100k_base'}"
            revision = resolved.revision
            limit = _pack_limit(resolved.effective_input_limit, resolved.pack_margin_ratio)
            settings = {
                "prefix": resolved.prefix,
                "suffix": resolved.suffix,
                "query_prefix": resolved.query_prefix,
                "query_suffix": resolved.query_suffix,
            }
            query_prefix = resolved.query_prefix
            query_suffix = resolved.query_suffix

        if limit <= 0:
            raise ValueError(f"Tokenizer configuration for '{model}' must include positive effective_input_limit")
        config = EmbeddingTokenizerConfig(
            identity=identity,
            revision=revision,
            effective_input_limit=limit,
            tier=tier,
            required_prefix=str(settings.get("prefix") or ""),
            required_suffix=str(settings.get("suffix") or ""),
            query_prefix=query_prefix,
            query_suffix=query_suffix,
        )
    except EmbeddingTokenizerUnavailableError:
        raise
    except Exception as exc:
        raise EmbeddingTokenizerUnavailableError(
            f"Tokenizer configuration is unavailable for embedding model '{model}'"
        ) from exc

    try:
        if tier == "local":
            from app.models.llm_server_client import get_local_embedding_model

            tokenizer = getattr(get_local_embedding_model(model).model, "tokenizer", None)
            if tokenizer is None:
                raise EmbeddingTokenizerUnavailableError(
                    f"Tokenizer is missing on the loaded local embedding model '{model}'"
                )

            def count_fn(formatted: str) -> int:
                return len(tokenizer(formatted, add_special_tokens=True, truncation=False)["input_ids"])

        elif tier == "exact":
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                config.identity, revision=config.revision, local_files_only=True
            )

            def count_fn(formatted: str) -> int:
                return len(tokenizer(formatted, add_special_tokens=True, truncation=False)["input_ids"])

        else:
            encoding_name = identity.split(":", 1)[-1] if identity.startswith("generic:") else "cl100k_base"
            count_fn = _tiktoken_counter(encoding_name)

        return config, _build_token_counter(config, count_fn)
    except EmbeddingTokenizerUnavailableError:
        raise
    except Exception as exc:
        raise EmbeddingTokenizerUnavailableError(
            f"Tokenizer '{config.identity}' is unavailable locally for embedding model '{model}'"
        ) from exc


__all__ = [
    "EmbeddingTokenizerConfig",
    "EmbeddingTokenizerUnavailableError",
    "resolve_embedding_tokenizer",
    "_split_oversized_fragment",
]