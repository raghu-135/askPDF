
"""
main.py - FastAPI entrypoint for the RAG Service

This module provides endpoints for document indexing, chat with retrieval-augmented generation, model listing, and health checks.

Endpoints:
- POST /index: Index a document for retrieval.
- POST /chat: Chat with RAG using LLM and embedding models.
- GET /status: Check if a collection exists in the vector DB.
- GET /models: List available LLM and embedding models from DMR.
- GET /health/model: Check if a specific model is ready.
- GET /health: Service health check.

Dependencies:
- FastAPI
- httpx
- dotenv
- rag, agent, vectordb.qdrant (local modules)
"""

import os
from dotenv import load_dotenv
load_dotenv()
from typing import List, Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import index_document
from agent import app as agent_app
from vectordb.qdrant import QdrantAdapter


app = FastAPI(title="RAG Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IndexRequest(BaseModel):
    """Request body for /index endpoint."""
    text: str
    embedding_model: str
    metadata: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""
    question: str
    llm_model: str
    embedding_model: str
    collection_name: Optional[str] = None
    history: List[Dict[str, str]] = []  # list of {role: "user"|"assistant", content: "..."}

@app.post("/index")
async def index_endpoint(req: IndexRequest):
    """
    Index a document for retrieval-augmented generation.
    Args:
        req (IndexRequest): Document text, embedding model, and optional metadata.
    Returns:
        Result of indexing operation.
    """
    try:
        result = await index_document(
            text=req.text,
            embedding_model_name=req.embedding_model,
            metadata=req.metadata
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Chat endpoint for retrieval-augmented generation.
    Args:
        req (ChatRequest): User question, LLM/embedding models, chat history, collection name.
    Returns:
        Answer and context from the agent.
    """
    try:
        from langchain_core.messages import HumanMessage, AIMessage
        chat_history = []
        for msg in req.history:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))

        inputs = {
            "question": req.question,
            "chat_history": chat_history,
            "llm_model": req.llm_model,
            "embedding_model": req.embedding_model,
            "collection_name": req.collection_name,
            "context": "",
            "answer": ""
        }

        result = await agent_app.ainvoke(inputs)
        return {"answer": result["answer"], "context": result["context"]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status_endpoint(collection_name: str):
    """
    Check if a collection exists and is ready in the vector database.
    Args:
        collection_name (str): Name of the collection to check.
    Returns:
        Status and collection name.
    """
    """Check if a collection exists and is ready"""
    try:
        db = QdrantAdapter()
        exists = await db.collection_exists(collection_name)
        return {"status": "ready" if exists else "not_ready", "collection": collection_name}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/models")
async def get_models():
    """
    Fetch available LLM and embedding models from DMR (Distributed Model Router).
    Returns:
        List of model IDs or fallback defaults if DMR is unavailable.
    """
    """Fetch available models from DMR"""
    dmr_url = os.getenv("DMR_BASE_URL")
    try:
        if not dmr_url.endswith("/v1"):
            dmr_url = f"{dmr_url}/v1"

        async with httpx.AsyncClient() as client:
            # Assuming standard OpenAI endpoint /v1/models
            resp = await client.get(f"{dmr_url}/models")
            if resp.status_code == 200:
                data = resp.json()
                print(f"DMR Models Found: {data}", flush=True)
                return data
            else:
                error_msg = f"DMR Fetch Failed {resp.status_code}: {resp.text}"
                print(error_msg, flush=True)
                raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Error fetching models from DMR: {str(e)}"
        print(error_msg, flush=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health/model")
async def model_health_endpoint(model: str):
    """
    Check if a specific model is available and ready.
    Args:
        model (str): Model ID to check.
    Returns:
        Model readiness status.
    """
    """Specific check for a model's availability"""
    from models import is_model_ready
    ready = await is_model_ready(model)
    return {"model": model, "ready": ready}

@app.get("/health")
async def health():
    """
    Service health check endpoint.
    Returns:
        Status OK if service is running.
    """
    return {"status": "ok"}
