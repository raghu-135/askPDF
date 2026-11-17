import json

import pytest

from app.services.embedding_model_catalog import reload_catalog
from app.services.embedding_tokenizer import EmbeddingTokenizerUnavailableError, resolve_embedding_tokenizer
from app.services.embedding_tokenizer_registry import (
    is_registered_embedding_model,
    lookup_embedding_tokenizer_settings,
    unique_tokenizer_identities,
)
from app.services.embedding_model_catalog import resolve_display_config, resolve_settings_for_provider


def test_builtin_openrouter_models_are_registered():
    qwen = lookup_embedding_tokenizer_settings("qwen/qwen3-embedding-8b")
    assert qwen["tokenizer"] == "Qwen/Qwen3-Embedding-8B"
    assert qwen["query_prefix"].startswith("Instruct:")
    e5 = lookup_embedding_tokenizer_settings("intfloat/e5-base-v2")
    assert e5["prefix"] == "passage: "
    assert e5["query_prefix"] == "query: "
    assert lookup_embedding_tokenizer_settings("liquid/lfm-2.5-embedding-350m:free")["tokenizer"] == (
        "LiquidAI/LFM2.5-Embedding-350M"
    )
    assert is_registered_embedding_model("sentence-transformers/all-minilm-l6-v2")
    assert not is_registered_embedding_model("openai/text-embedding-3-small")


def test_catalog_path_override(tmp_path, monkeypatch):
    catalog = {
        "generic": {"encoding": "cl100k_base", "effective_input_limit": 8192, "pack_margin_ratio": 0.85},
        "models": {
            "qwen/qwen3-embedding-8b": {
                "tokenizer": "custom/tok",
                "effective_input_limit": 128,
            }
        },
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    monkeypatch.setenv("EMBEDDING_MODELS_CONFIG", str(path))
    reload_catalog()
    settings = lookup_embedding_tokenizer_settings("qwen/qwen3-embedding-8b")
    assert settings["tokenizer"] == "custom/tok"
    assert settings["effective_input_limit"] == 128
    reload_catalog()


def test_unique_tokenizer_identities_dedupe_qwen_and_nomic_aliases():
    identities = dict(unique_tokenizer_identities())
    assert "Qwen/Qwen3-Embedding-8B" in identities
    assert "nomic-ai/nomic-embed-text-v1.5" in identities
    assert len(identities) == len(set(identities))


def test_unknown_remote_model_uses_generic_tiktoken(monkeypatch):
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "model-a")
    resolve_embedding_tokenizer.cache_clear()
    config, counter = resolve_embedding_tokenizer("openai/text-embedding-3-small")
    assert config.tier == "generic"
    assert config.identity.startswith("generic:")
    assert counter.count("hello world") > 0


def test_generic_display_config():
    resolved = resolve_display_config("openai/text-embedding-3-small")
    assert resolved.tier == "generic"
    assert resolved.encoding == "cl100k_base"


def test_exact_tier_settings():
    resolved = resolve_settings_for_provider("intfloat/e5-base-v2")
    assert resolved.tier == "exact"
    assert resolved.tokenizer == "intfloat/e5-base-v2"