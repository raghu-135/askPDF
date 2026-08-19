"""Deterministic OpenAI-compatible provider for external runtime CI."""

import asyncio
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {"id": "phase5-deterministic", "object": "model"},
            {"id": "phase5-deterministic-embedding", "object": "model"},
        ],
    }


@app.post("/v1/embeddings")
async def embeddings(payload: dict):
    values = payload.get("input")
    inputs = values if isinstance(values, list) else [values]
    data = []
    for index, value in enumerate(inputs):
        text = str(value or "")
        # Stable, non-zero vectors keep the boundary test deterministic without
        # downloading or initializing a Hugging Face embedding model.
        checksum = sum(text.encode("utf-8")) or 1
        vector = [((checksum + offset * 17) % 101 + 1) / 101 for offset in range(8)]
        data.append({"object": "embedding", "index": index, "embedding": vector})
    return {
        "object": "list",
        "model": payload.get("model") or "phase5-deterministic-embedding",
        "data": data,
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


@app.post("/v1/chat/completions")
async def completions(payload: dict):
    messages = payload.get("messages") or []
    question = next((str(item.get("content")) for item in reversed(messages) if item.get("role") == "user"), "")
    prompt = "\n".join(str(item.get("content") or "") for item in messages).lower()
    content = "Deterministic evidence summary: the fake provider confirms the requested document evidence."
    if "strict router for a rag workflow" in prompt:
        content = '{"route":"direct","reason":"Deterministic boundary smoke route."}'
    elif "strict answer-quality evaluator" in prompt:
        content = '{"pass":true,"reason":"Deterministic answer accepted.","issues":[]}'
    elif "clarif" in question.lower():
        content = "Deterministic clarification response."
    response = {
        "id": "phase5-deterministic-response",
        "object": "chat.completion",
        "model": payload.get("model") or "phase5-deterministic",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    if not payload.get("stream"):
        return response

    async def chunks():
        words = content.split(" ")
        slow = "continue until stopped" in question.lower()
        for index, word in enumerate(words):
            chunk = {
                "id": response["id"],
                "object": "chat.completion.chunk",
                "model": response["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": (("" if index == 0 else " ") + word)},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            if slow:
                await asyncio.sleep(1)
        final = {
            "id": response["id"],
            "object": "chat.completion.chunk",
            "model": response["model"],
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n"

    return StreamingResponse(chunks(), media_type="text/event-stream")
