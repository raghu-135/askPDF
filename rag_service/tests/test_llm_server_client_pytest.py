"""Tests for LLM client response normalization shims."""



from app.agent.reasoning import normalize_ai_response
import pytest

from app.models.llm_server_client import (
    ReasoningChatOpenAI,
    close_model_client,
    get_llm,
)


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
