"""
Models API Module - Model and health check endpoints.

Endpoints:
- GET /api/models - List available models
- GET /api/health/chat-model/{model} - Check chat model status
- GET /api/health/embed-model/{model} - Check embedding model status
"""

import traceback

from fastapi import APIRouter, HTTPException

from app.models.llm_server_client import (
    check_chat_model_ready,
    check_embed_model_ready,
    check_model_supports_tools,
    fetch_available_models,
)

router = APIRouter(tags=["models"])


@router.get("/models")
async def get_models():
    """Fetch available LLM and embedding models from the LLM API/server."""
    try:
        return await fetch_available_models()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/chat-model/{model:path}")
async def is_chat_model_ready_endpoint(model: str):
    """
    Check if a chat/LLM model is ready AND supports tool calling.
    Returns chat_model_ready and supports_tools as separate flags.
    """
    try:
        ready = await check_chat_model_ready(model)
        if not ready:
            return {"model": model, "chat_model_ready": False, "supports_tools": False}
        supports_tools = await check_model_supports_tools(model)
        return {"model": model, "chat_model_ready": True, "supports_tools": supports_tools}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/embed-model/{model:path}")
async def is_embed_model_ready_endpoint(model: str):
    """Check if an embedding model is ready."""
    try:
        ready = await check_embed_model_ready(model)
        return {"model": model, "embed_model_ready": ready}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
