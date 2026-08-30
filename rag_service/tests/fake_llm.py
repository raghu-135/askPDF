"""Deterministic OpenAI-compatible provider for external runtime CI."""

from fastapi import FastAPI

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
    return {
        "id": "phase5-deterministic-response",
        "object": "chat.completion",
        "model": payload.get("model") or "phase5-deterministic",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
