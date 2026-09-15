import pytest

from app.services.embedding_tokenizer import EmbeddingTokenizerUnavailableError, resolve_embedding_tokenizer
from app.services.embedding_tokenizer_registry import (
    is_registered_embedding_model,
    lookup_embedding_tokenizer_settings,
    unique_tokenizer_identities,
)


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


def test_env_tokenizer_json_overrides_builtin(monkeypatch):
    monkeypatch.setenv(
        "EMBEDDING_TOKENIZER_CONFIG_JSON",
        '{"qwen/qwen3-embedding-8b":{"tokenizer":"custom/tok","effective_input_limit":128}}',
    )
    settings = lookup_embedding_tokenizer_settings("qwen/qwen3-embedding-8b")
    assert settings["tokenizer"] == "custom/tok"
    assert settings["effective_input_limit"] == 128


def test_unique_tokenizer_identities_dedupe_qwen_and_nomic_aliases():
    identities = dict(unique_tokenizer_identities())
    assert "Qwen/Qwen3-Embedding-8B" in identities
    assert "nomic-ai/nomic-embed-text-v1.5" in identities
    assert len(identities) == len(set(identities))


def test_unknown_remote_model_still_requires_registration(monkeypatch):
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "model-a")
    monkeypatch.setenv("EMBEDDING_TOKENIZER_CONFIG_JSON", "")
    resolve_embedding_tokenizer.cache_clear()
    with pytest.raises(EmbeddingTokenizerUnavailableError):
        resolve_embedding_tokenizer("openai/text-embedding-3-small")
