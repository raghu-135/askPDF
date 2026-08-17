"""Deterministic OpenAI-compatible provider for external runtime CI."""

from fastapi import FastAPI

app = FastAPI()


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "phase5-deterministic", "object": "model"}]}


@app.post("/v1/chat/completions")
async def completions(payload: dict):
    messages = payload.get("messages") or []
    question = next((str(item.get("content")) for item in reversed(messages) if item.get("role") == "user"), "")
    content = "Deterministic evidence summary: the fake provider confirms the requested document evidence."
    if "clarif" in question.lower():
        content = "Deterministic clarification response."
    return {
        "id": "phase5-deterministic-response",
        "object": "chat.completion",
        "model": payload.get("model") or "phase5-deterministic",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
