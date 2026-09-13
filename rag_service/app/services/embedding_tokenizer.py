"""Embedding tokenizer configuration and exact input-budget accounting."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

from app.services.document_pipeline import TokenCounter


@dataclass(frozen=True)
class EmbeddingTokenizerConfig:
    identity: str
    revision: str | None
    effective_input_limit: int
    required_prefix: str = ""
    required_suffix: str = ""

    @property
    def fingerprint(self) -> str:
        return f"{self.identity}@{self.revision or 'default'}:{self.effective_input_limit}:{self.required_prefix}:{self.required_suffix}"


def _external_mapping() -> dict[str, Any]:
    raw = os.environ.get("EMBEDDING_TOKENIZER_CONFIG_JSON", "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("EMBEDDING_TOKENIZER_CONFIG_JSON must be valid JSON") from exc
    return value if isinstance(value, dict) else {}


@lru_cache(maxsize=32)
def resolve_embedding_tokenizer(model: str) -> tuple[EmbeddingTokenizerConfig, TokenCounter]:
    """Resolve the actual local tokenizer or reject an unknown external model."""
    model = str(model or "").strip()
    if not model:
        raise ValueError("embedding model is required")

    local_model = os.environ.get("LOCAL_EMBEDDING_MODEL", "").strip()
    settings: dict[str, Any]
    if model == local_model:
        settings = {
            "tokenizer": os.environ.get("LOCAL_EMBEDDING_TOKENIZER", model),
            "revision": os.environ.get("LOCAL_EMBEDDING_TOKENIZER_REVISION") or None,
            "effective_input_limit": int(os.environ.get("LOCAL_EMBEDDING_INPUT_LIMIT", "512")),
            "prefix": os.environ.get("LOCAL_EMBEDDING_FORMAT_PREFIX", ""),
            "suffix": os.environ.get("LOCAL_EMBEDDING_FORMAT_SUFFIX", ""),
        }
    else:
        settings = _external_mapping().get(model) or {}
        if not settings:
            raise ValueError(
                f"No tokenizer configuration is registered for external embedding model '{model}'. "
                "Set EMBEDDING_TOKENIZER_CONFIG_JSON with tokenizer, effective_input_limit, and formatting."
            )

    identity = str(settings.get("tokenizer") or settings.get("identity") or "").strip()
    limit = int(settings.get("effective_input_limit") or settings.get("input_limit") or 0)
    if not identity or limit <= 0:
        raise ValueError(f"Tokenizer configuration for '{model}' must include tokenizer and positive effective_input_limit")
    config = EmbeddingTokenizerConfig(
        identity=identity,
        revision=str(settings.get("revision") or "").strip() or None,
        effective_input_limit=limit,
        required_prefix=str(settings.get("prefix") or ""),
        required_suffix=str(settings.get("suffix") or ""),
    )

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.identity, revision=config.revision, local_files_only=True)

        def count(text: str) -> int:
            formatted = f"{config.required_prefix}{text}{config.required_suffix}"
            return len(tokenizer(formatted, add_special_tokens=True, truncation=False)["input_ids"])

        def split(text: str, limit: int) -> Sequence[str]:
            words = text.split()
            output: list[str] = []
            current: list[str] = []
            for word in words:
                candidate = " ".join([*current, word])
                if current and count(candidate) > limit:
                    output.append(" ".join(current))
                    current = [word]
                else:
                    current.append(word)
            if current:
                output.append(" ".join(current))
            return output

        return config, TokenCounter(count=count, split=split)
    except Exception as exc:
        if os.environ.get("EMBEDDING_TOKENIZER_ALLOW_FALLBACK", "false").lower() != "true":
            raise ValueError(
                f"Tokenizer '{config.identity}' is unavailable locally for embedding model '{model}'. "
                "Install/cache the tokenizer or explicitly set EMBEDDING_TOKENIZER_ALLOW_FALLBACK=true only for development tests."
            ) from exc
        words = lambda text: len(str(text).split())
        return config, TokenCounter(
            count=words,
            split=lambda text, limit: [
                " ".join(str(text).split()[i:i + max(1, limit)])
                for i in range(0, len(str(text).split()), max(1, limit))
            ],
        )


__all__ = ["EmbeddingTokenizerConfig", "resolve_embedding_tokenizer"]
