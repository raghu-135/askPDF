"""Tests for LLM client response normalization shims."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agent.reasoning import normalize_ai_response
from app.models.llm_server_client import ReasoningChatOpenAI


def test_reasoning_chat_openai_preserves_lm_studio_reasoning_content():
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
