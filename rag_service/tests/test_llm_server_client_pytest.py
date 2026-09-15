"""Tests for LLM client response normalization shims."""



from app.agent.reasoning import normalize_ai_response
import httpx
import pytest

from app.models.llm_server_client import (
    LOCAL_EMBEDDING_MODELS,
    ReasoningChatOpenAI,
    _chat_probe_accepted,
    _congested_provider_status,
    _embedding_probe_accepted,
    _model_ready_cache,
    check_chat_model_ready,
    check_embedding_model_ready,
    close_model_client,
    fetch_available_models,
    get_llm,
    llm_provider_auth,
    openai_sdk_default_headers,
)


def test_provider_auth_sends_bearer_token_when_required(monkeypatch):
    monkeypatch.setenv("LLM_API_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-or-test-key")
    api_key, headers = llm_provider_auth()
    assert api_key == "sk-or-test-key"
    assert headers == {"Authorization": "Bearer sk-or-test-key"}


def test_provider_auth_omits_authorization_for_keyless_local_providers(monkeypatch):
    monkeypatch.setenv("LLM_API_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    api_key, headers = llm_provider_auth()
    assert api_key == "not-needed"
    assert headers == {}


def test_openai_sdk_headers_drop_authorization_to_avoid_cloudflare_400():
    assert openai_sdk_default_headers({"Authorization": "Bearer sk-or-test-key"}) is None
    assert openai_sdk_default_headers({"authorization": "Bearer sk-or-test-key", "X-Title": "askPDF"}) == {
        "X-Title": "askPDF"
    }


def test_openrouter_rate_limit_is_congestion_not_missing_model():
    assert _congested_provider_status(429) is True
    assert _congested_provider_status(502) is True
    assert _congested_provider_status(503) is True
    assert _congested_provider_status(403) is False
    assert _congested_provider_status(404) is False


@pytest.mark.asyncio
async def test_chat_readiness_treats_gemma_free_rate_limit_as_ready(monkeypatch):
    monkeypatch.setenv("LLM_API_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model_id = "google/gemma-4-31b-it:free"
    _model_ready_cache.clear()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": model_id}]})
        return httpx.Response(429, json={"error": {"message": "Rate limited"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("app.http_clients.get_http_client", lambda _name: client)
    assert await check_chat_model_ready(model_id) is True


@pytest.mark.asyncio
async def test_chat_readiness_probes_only_on_cache_miss(monkeypatch):
    monkeypatch.setenv("LLM_API_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model_id = "google/gemma-4-31b-it:free"
    _model_ready_cache.clear()
    chat_posts = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": model_id}]})
        chat_posts.append(request.url.path)
        return httpx.Response(
            200,
            json={"model": model_id, "choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("app.http_clients.get_http_client", lambda _name: client)
    assert await check_chat_model_ready(model_id) is True
    assert await check_chat_model_ready(model_id) is True
    assert len(chat_posts) == 1


def test_chat_probe_accepts_openrouter_alias_resolution_and_empty_first_token():
    assert _chat_probe_accepted(
        "~deepseek/deepseek-v4-flash-latest",
        {
            "model": "deepseek/deepseek-v4-flash-0731",
            "choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "length"}],
        },
    )
    assert _chat_probe_accepted(
        "qwen/qwen3.8-27b",
        {"model": "qwen/qwen3.8-27b", "choices": [{"message": {"role": "assistant", "content": ""}}]},
    )
    assert not _chat_probe_accepted("qwen/qwen3.8-27b", {"model": "qwen/qwen3.8-27b", "choices": []})


@pytest.mark.asyncio
async def test_reasoning_chat_openai_preserves_lm_studio_reasoning_content():
    llm = ReasoningChatOpenAI(
        model="deepseek/deepseek-r1-0528-qwen3-8b",
        base_url="http://localhost:1234/v1",
        api_key="sk-no-key-required",
    )
    chat_result = llm._create_chat_result(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "deepseek/deepseek-r1-0528-qwen3-8b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Final answer",
                        "reasoning_content": "LM Studio reasoning trace",
                        "tool_calls": [],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "completion_tokens_details": {
                    "reasoning_tokens": 263,
                }
            },
        }
    )

    message = chat_result.generations[0].message
    normalized = normalize_ai_response(message)

    assert message.additional_kwargs["reasoning_content"] == "LM Studio reasoning trace"
    assert normalized["answer"] == "Final answer"
    assert normalized["reasoning"] == "LM Studio reasoning trace"
    assert normalized["reasoning_available"] is True
    assert normalized["reasoning_format"] == "structured"
    await close_model_client(llm)


@pytest.mark.asyncio
async def test_closing_owned_llm_transport_does_not_close_another_wrapper():
    first = get_llm("test-model", own_async_transport=True)
    second = get_llm("test-model", own_async_transport=True)

    assert first.http_async_client is not second.http_async_client
    assert not first.http_async_client.is_closed
    assert not second.http_async_client.is_closed

    await close_model_client(first)

    assert first.http_async_client.is_closed
    assert not second.http_async_client.is_closed
    await close_model_client(second)


@pytest.mark.asyncio
async def test_closing_implicit_llm_wrapper_does_not_close_shared_transport():
    first = get_llm("test-model")
    second = get_llm("test-model")
    shared_transport = first.root_async_client._client

    assert shared_transport is second.root_async_client._client
    await close_model_client(first)

    assert not shared_transport.is_closed


@pytest.mark.asyncio
async def test_openrouter_model_list_includes_registered_embedding_catalog(monkeypatch):
    monkeypatch.setenv("LLM_API_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/embeddings/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "qwen/qwen3-embedding-8b"},
                        {"id": "sentence-transformers/all-minilm-l6-v2"},
                        {"id": "openai/text-embedding-3-small"},
                    ]
                },
            )
        if path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "openai/gpt-4o-mini"}]})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("app.http_clients.get_http_client", lambda _name: client)
    result = await fetch_available_models()
    assert result["embedding_models"] == [
        "qwen/qwen3-embedding-8b",
        "sentence-transformers/all-minilm-l6-v2",
    ]
    assert "openai/text-embedding-3-small" not in result["embedding_models"]
    assert result["local_embedding_models"] == LOCAL_EMBEDDING_MODELS
    assert "openai/gpt-4o-mini" in result["llm_models"]


@pytest.mark.asyncio
async def test_openrouter_embedding_readiness_uses_embeddings_catalog(monkeypatch):
    monkeypatch.setenv("LLM_API_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model_id = "qwen/qwen3-embedding-4b"
    _model_ready_cache.clear()
    embed_posts = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/embeddings/models"):
            return httpx.Response(200, json={"data": [{"id": model_id}]})
        if path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "openai/gpt-4o-mini"}]})
        if request.method == "POST" and path.endswith("/embeddings"):
            embed_posts.append(path)
            return httpx.Response(
                200,
                json={"model": model_id, "data": [{"embedding": [0.1, 0.2], "index": 0}]},
            )
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("app.http_clients.get_http_client", lambda _name: client)
    assert await check_embedding_model_ready(model_id) is True
    assert embed_posts == ["/api/v1/embeddings"]


def test_embedding_probe_accepts_huggingface_casing():
    assert _embedding_probe_accepted(
        "qwen/qwen3-embedding-4b",
        {"model": "Qwen/Qwen3-Embedding-4B", "data": [{"embedding": [0.1], "index": 0}]},
    )
    assert not _embedding_probe_accepted(
        "qwen/qwen3-embedding-4b",
        {"model": "Qwen/Qwen3-Embedding-4B", "data": []},
    )


@pytest.mark.asyncio
async def test_openrouter_embedding_readiness_accepts_huggingface_model_echo(monkeypatch):
    monkeypatch.setenv("LLM_API_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model_id = "qwen/qwen3-embedding-4b"
    _model_ready_cache.clear()

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/embeddings/models"):
            return httpx.Response(200, json={"data": [{"id": model_id}]})
        if path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "openai/gpt-4o-mini"}]})
        if request.method == "POST" and path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "model": "Qwen/Qwen3-Embedding-4B",
                    "data": [{"embedding": [0.1, 0.2], "index": 0}],
                },
            )
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("app.http_clients.get_http_client", lambda _name: client)
    assert await check_embedding_model_ready(model_id) is True
