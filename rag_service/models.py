"""
Model utilities for LLM API/OpenAI-compatible APIs.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import httpx
import asyncio
import logging
from fastapi import HTTPException
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_base_url() -> str:
    """Get the LLM API base URL, ensuring it ends with /v1."""
    base_url = os.getenv("LLM_API_URL")
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"

async def fetch_available_models():
    """
    Fetch available models from the LLM API/server (OpenAI-compatible) and categorize as embedding, llm, or unknown.
    Returns:
        dict: {"embedding_models": [...], "llm_models": [...], "unknown_models": [...], "all_models": [...]}
    """
    llm_api_url = os.getenv("LLM_API_URL")
    try:
        if not llm_api_url.endswith("/v1"):
            llm_api_url = f"{llm_api_url}/v1"

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{llm_api_url}/models")
            if resp.status_code == 200:
                data = resp.json()
                # OpenAI-compatible: models are in data['data']
                models = data.get('data', []) if isinstance(data, dict) else data
                model_ids = [m['id'] if isinstance(m, dict) and 'id' in m else m for m in models]
                embedding_models = [m for m in model_ids if is_embedding_model_by_keyword(m)]
                llm_models = [m for m in model_ids if is_llm_model_by_keyword(m)]
                not_embedding_models = [m for m in model_ids if m not in embedding_models]
                not_llm_models = [m for m in model_ids if m not in llm_models]
                result = {
                    "embedding_models": embedding_models,
                    "llm_models": llm_models,
                    "all_models": model_ids,
                    "not_embedding_models": not_embedding_models,
                    "not_llm_models": not_llm_models
                }
                logger.info(f"LLM API/server Models Response: {result}")
                return result
            else:
                error_msg = f"LLM API/server Fetch Failed {resp.status_code}: {resp.text}"
                print(error_msg, flush=True)
                raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Error fetching models from LLM API/server: {str(e)}"
        print(error_msg, flush=True)
        raise HTTPException(status_code=500, detail=error_msg)

# Identify embedding models by keywords in model id
def is_embedding_model_by_keyword(model_id: str) -> bool:
    """
    Returns True if the model id suggests it is an embedding model (by keyword).
    """
    name = model_id.lower()
    keywords = ["embed", "bge", "gte", "e5", "sts", "query", "passage"]
    return any(k in name for k in keywords)

# Identify LLM models by keywords in model id
def is_llm_model_by_keyword(model_id: str) -> bool:
    """
    Returns True if the model id suggests it is an LLM/chat/completion model (by keyword).
    """
    name = model_id.lower()
    # Exclude embedding models
    if is_embedding_model_by_keyword(name):
        return False
    keywords = ["chat", "instruct", "completion", "base", "llama", "mistral", "qwen", "deepseek", "vicuna", "falcon", "gpt", "codellama", "phi", "mixtral", "yi", "zephyr", "dbrx", "command", "orca", "hermes", "openchat", "wizard", "llava", "starling", "solar"]
    return any(k in name for k in keywords)

def get_llm(model_name: str, temperature: float = 0.0):
    """
    Return a configured ChatOpenAI client for the given model.
    """
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url=_get_base_url(),
        api_key="sk-no-key-required"
    )

def get_embedding_model(model_name: str):
    """
    Return a configured OpenAIEmbeddings client for the given model.
    """
    return OpenAIEmbeddings(
        model=model_name,
        base_url=_get_base_url(),
        api_key="sk-no-key-required",
        check_embedding_ctx_length=False
    )

def get_system_prompt(context: str, use_history: bool = False, use_web: bool = False) -> str:
    """
    Constructs a consistent system prompt for the AI assistant across different service nodes.
    """
    sources = ["PDF context"]
    if use_web:
        sources.append("Web Search results")
    if use_history:
        sources.append("past conversations")
    
    sources_str = " and ".join(sources)

    instructions = [
        "Answer in natural language. Do NOT output JSON, code blocks for tools, or function calls.",
    ]
    
    if use_history:
        instructions.append("If the context contains relevant information from past conversations, incorporate it into your answer and acknowledge it naturally (e.g., 'As we discussed before...').")
    
    if use_web:
        instructions.append("If the web search failed, rely on the PDF context.")
        
    instructions.append("If the answer is not in the context and you cannot answer it, state that you don't know based on the provided information.")
    
    formatted_instructions = "\n".join([f"{i+1}. {instr}" for i, instr in enumerate(instructions)])
    
    return (
        f"You are a helpful AI assistant. Use the provided {sources_str} to answer the user's question accurately.\n\n"
        f"INSTRUCTIONS:\n{formatted_instructions}\n\n"
        f"CONTEXT:\n{context}"
    )

async def check_chat_model_ready(model_name: str) -> bool:
    """
    Check if the supplied model is a chat model and is ready in the LLM API/server.
    Returns True if ready, False if not ready or not found.
    """
    base_url = _get_base_url()
    try:
        async with httpx.AsyncClient() as client:
            # 1. Check if model exists
            resp = await client.get(f"{base_url}/models", timeout=5.0)
            if resp.status_code != 200:
                return False
            models = resp.json().get('data', [])
            model_ids = [m['id'] for m in models]
            if model_name not in model_ids:
                return False

            # 2. Probe with Chat Completion
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1
            }
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    chat_resp = await client.post(f"{base_url}/chat/completions", json=payload, timeout=30.0)
                    logger.info(f"Chat completion probe response status: {chat_resp.status_code}")
                    if chat_resp.status_code == 200:
                        return True
                    if chat_resp.status_code == 503 and attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # exponential backoff
                        continue
                    if chat_resp.status_code == 503:
                        return True
                    if chat_resp.status_code == 500:
                        return False
                    break
                except httpx.ReadTimeout:
                    logger.warning(f"Timeout on attempt {attempt + 1} checking chat model {model_name}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return False
                except Exception as e:
                    logging.exception("Exception during chat completion probe : %s", e)
                    return False
                
            # 3. Fallback: If model is listed and not 503, assume reachable
            return True

    except Exception:
        logging.exception("Exception during model readiness check")
        return False

async def check_embed_model_ready(model_name: str) -> bool:
    """
    Check if the supplied model is an embedding model and is ready in the LLM API/server.
    Returns True if ready, False if not ready or not found.
    """
    base_url = _get_base_url()
    try:
        async with httpx.AsyncClient() as client:
            # 1. Check if model exists
            resp = await client.get(f"{base_url}/models", timeout=5.0)
            if resp.status_code != 200:
                return False
            models = resp.json().get('data', [])
            model_ids = [m['id'] for m in models]
            if model_name not in model_ids:
                return False

            # 2. Probe with Embeddings
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    emb_resp = await client.post(
                        f"{base_url}/embeddings",
                        json={"model": model_name, "input": "hi"},
                        timeout=30.0
                    )
                    if emb_resp.status_code == 200:
                        return True
                    if emb_resp.status_code == 503 and attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # exponential backoff
                        continue
                    if emb_resp.status_code == 503:
                        return False
                    break
                except httpx.ReadTimeout:
                    logger.warning(f"Timeout on attempt {attempt + 1} checking embedding model {model_name}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return False
                except Exception as e:
                    logging.exception("Exception during embedding probe : %s", e)
                    return False

            # 3. Fallback: If model is listed and not 200 or 503, assume not reachable
            return False
    except Exception:
        logging.exception("Exception during embedding model readiness check")
        return False
