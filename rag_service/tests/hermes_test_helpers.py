from __future__ import annotations

import json
import os
from typing import Any

import httpx


RUNTIME_URL = os.getenv("HERMES_RUNTIME_URL", "http://hermes-runtime:8200")


def runtime_payload(run_id: str, question: str = "deterministic proof") -> dict[str, Any]:
    return {
        "request": {
            "run_id": run_id,
            "thread_id": f"thread-{run_id}",
            "definition_id": "hermes_rag_agent",
            "framework": "hermes",
            "builder_id": "hermes_agent",
            "input": {"question": question},
            "options": {"llm_model": "phase5-deterministic", "llm_provider": "custom"},
        },
        "context": {
            "resolved_spec": {
                "definition_version": 1,
                "config": {
                    "mcp_server": "askpdf",
                    "allowed_tool_ids": ["document_evidence", "clarify_intent"],
                    "system_prompt": "Use approved tools.",
                    "model": "phase5-deterministic",
                    "provider": "custom",
                },
            },
        },
    }


async def read_sse(client: httpx.AsyncClient, method: str, path: str, **kwargs: Any) -> list[dict[str, Any]]:
    async with client.stream(method, path, **kwargs) as response:
        assert response.status_code == 200, await response.aread()
        body = await response.aread()
    values = []
    for block in body.decode().split("\n\n"):
        line = next((line for line in block.splitlines() if line.startswith("data:")), None)
        if line:
            values.append(json.loads(line[5:].strip()))
    return values
