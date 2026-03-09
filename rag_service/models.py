import os
from dotenv import load_dotenv
load_dotenv()
import httpx
import asyncio
import logging
import time
from typing import Dict, Tuple
from fastapi import HTTPException
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for model readiness checks
_model_ready_cache: Dict[str, Tuple[bool, float]] = {}
CACHE_TTL = 300  # 5 minutes for successful checks
CACHE_TTL_FAIL = 15  # 15 seconds for failed checks

def _check_model_ready_cache(cache_key: str) -> bool | None:
    """Returns the cached readiness status if valid, otherwise None."""
    cached = _model_ready_cache.get(cache_key)
    if cached:
        is_ready, timestamp = cached
        ttl = CACHE_TTL if is_ready else CACHE_TTL_FAIL
        if time.time() - timestamp < ttl:
            return is_ready
    return None

def _update_model_ready_cache(cache_key: str, is_ready: bool) -> bool:
    """Updates the cache and returns the readiness status."""
    _model_ready_cache[cache_key] = (is_ready, time.time())
    return is_ready


# Token budget configuration
def get_default_token_budget():
    budget = os.getenv("DEFAULT_TOKEN_BUDGET")
    if budget is None:
        raise ValueError("DEFAULT_TOKEN_BUDGET environment variable is not set")
    return int(budget)

def get_default_max_iterations():
    val = os.getenv("DEFAULT_MAX_ITERATIONS")
    if val is None:
        raise ValueError("DEFAULT_MAX_ITERATIONS environment variable is not set")
    return int(val)

def get_min_max_iterations():
    val = os.getenv("MIN_MAX_ITERATIONS")
    if val is None:
        raise ValueError("MIN_MAX_ITERATIONS environment variable is not set")
    return int(val)

def get_max_max_iterations():
    val = os.getenv("MAX_MAX_ITERATIONS")
    if val is None:
        raise ValueError("MAX_MAX_ITERATIONS environment variable is not set")
    return int(val)

DEFAULT_TOKEN_BUDGET = get_default_token_budget()
DEFAULT_MAX_ITERATIONS = get_default_max_iterations()
MIN_MAX_ITERATIONS = get_min_max_iterations()
MAX_MAX_ITERATIONS = get_max_max_iterations()

def get_env_int(name: str, default: int | None = None) -> int:
    val = os.getenv(name)
    if val is None:
        if default is not None:
            return default
        raise ValueError(f"{name} environment variable is not set")
    return int(val)

MAX_CUSTOM_INSTRUCTIONS_CHARS = get_env_int("MAX_CUSTOM_INSTRUCTIONS_CHARS")
MAX_SYSTEM_ROLE_CHARS = get_env_int("MAX_SYSTEM_ROLE_CHARS")
MAX_TOOL_INSTRUCTION_CHARS = get_env_int("MAX_TOOL_INSTRUCTION_CHARS")

INTENT_AGENT_MAX_ITERATIONS = get_env_int("INTENT_AGENT_MAX_ITERATIONS", default=1)
MAX_ITERATIONS_SUFFICIENT_COVERAGE = get_env_int("MAX_ITERATIONS_SUFFICIENT_COVERAGE", default=2)
MAX_ITERATIONS_PROBABLY_SUFFICIENT_COVERAGE = get_env_int("MAX_ITERATIONS_PROBABLY_SUFFICIENT_COVERAGE", default=4)
WEB_SEARCH_ITERATION_BONUS = get_env_int("WEB_SEARCH_ITERATION_BONUS", default=2)

# Context allocation ratios (must sum to 1.0)
RATIO_LLM_RESPONSE = 0.25      # Reserve 25% for answer
RATIO_DOCUMENT_CONTEXT = 0.45       # 45% for document chunks (PDF + webpage)
RATIO_SEMANTIC_MEMORY = 0.30   # 30% for recalled semantic memories

# Individual item limits
RATIO_MEMORY_SUMMARIZATION_THRESHOLD = 0.05  # Summarize if > 5% of total window
RATIO_MEMORY_HARD_LIMIT = 0.10               # Truncate if > 10% of total window (fallback)

# Chars per token estimate
CHARS_PER_TOKEN = 4

# Pre-fetch budget allocation ratios
# These three sum to 0.68; the remaining 0.32 is reserved for answer generation +
# system prompt overhead (tool schemas, locked sections, etc.)
RATIO_PREFETCH_RECENT = 0.22    # Recent verbatim conversation turns injected inline
RATIO_PREFETCH_SEMANTIC = 0.18  # Semantic chat-memory recall from all past QA pairs
RATIO_PREFETCH_DOCUMENT = 0.28       # Document evidence (top-K chunks, raw question query)

# Average char estimates used to derive item-count limits from char budgets
AVG_CHUNK_CHARS = 500   # Typical PDF or chat-memory chunk
AVG_TURN_CHARS = 600    # Typical combined user+assistant turn


def compute_prefetch_budget(context_window: int) -> dict:
    """
    Compute character and item-count budgets for the parallel pre-fetch pass.

    Scales proportionally to any context window (4 K → 1 M tokens).
    A 20 % overhead buffer is preserved for system prompt text + tool schemas
    injected by the LangChain / LangGraph framework.

    Returns a dict with both char budgets and derived item-count limits so
    callers can use whichever unit is most convenient.
    """
    usable = int(context_window * 0.80 * CHARS_PER_TOKEN)
    return {
        # Character budgets
        "recent_history_chars":   int(usable * RATIO_PREFETCH_RECENT),
        "semantic_history_chars": int(usable * RATIO_PREFETCH_SEMANTIC),
        "document_context_chars":      int(usable * RATIO_PREFETCH_DOCUMENT),
        # Derived item-count limits
        "document_limit":              max(3, int(usable * RATIO_PREFETCH_DOCUMENT)      // AVG_CHUNK_CHARS),
        "semantic_limit":         max(3, int(usable * RATIO_PREFETCH_SEMANTIC) // AVG_CHUNK_CHARS),
        "recent_turn_limit":      max(4, int(usable * RATIO_PREFETCH_RECENT)   // AVG_TURN_CHARS),
    }


def default_thread_settings():
    """Default persisted settings for a thread."""
    return {
        "max_iterations": DEFAULT_MAX_ITERATIONS,
        "min_max_iterations": MIN_MAX_ITERATIONS,
        "max_max_iterations": MAX_MAX_ITERATIONS,
        "context_window": DEFAULT_TOKEN_BUDGET,
        "system_role": "Expert AI Research Assistant specializing in analyzing uploaded documents and synthesizing accurate answers.",
        "tool_instructions": {},
        "custom_instructions": "",
        "use_intent_agent": True,
        "intent_agent_max_iterations": INTENT_AGENT_MAX_ITERATIONS,
        "reasoning_mode": True,
    }


def merge_thread_settings(overrides=None):
    """Merge arbitrary overrides onto defaults while preserving known keys."""
    merged = default_thread_settings()
    if isinstance(overrides, dict):
        merged.update({k: overrides.get(k) for k in merged.keys() if k in overrides})
    return merged


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
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Error fetching models from LLM API/server: {str(e)}"
        logger.error(error_msg)
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
    sources = ["document context"]
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
        instructions.append("If the web search failed, rely on the document context.")
        
    instructions.append("If the answer is not in the context and you cannot answer it, state that you don't know based on the provided information.")
    
    formatted_instructions = "\n".join([f"{i+1}. {instr}" for i, instr in enumerate(instructions)])
    
    return (
        f"You are a helpful AI assistant. Use the provided {sources_str} to answer the user's question accurately.\n\n"
        f"INSTRUCTIONS:\n{formatted_instructions}\n\n"
        f"CONTEXT:\n{context}"
    )

async def _check_model_exists(client: httpx.AsyncClient, base_url: str, model_name: str) -> bool:
    """Helper to check if a model ID exists in the /models endpoint."""
    try:
        resp = await client.get(f"{base_url}/models", timeout=15.0)
        if resp.status_code != 200:
            return False
        data = resp.json()
        models = data.get('data', []) if isinstance(data, dict) else data
        model_ids = [m['id'] if isinstance(m, dict) and 'id' in m else m for m in models]
        return model_name in model_ids
    except Exception:
        return False

async def _probe_with_retry(
    client: httpx.AsyncClient, 
    url: str, 
    payload: dict, 
    validator, 
    model_name: str, 
    probe_type: str,
    max_retries: int = 3
) -> bool:
    """Helper to probe an endpoint with retry logic and exponential backoff."""
    for attempt in range(max_retries):
        try:
            resp = await client.post(url, json=payload, timeout=30.0)
            logger.info(f"{probe_type} probe response status for {model_name}: {resp.status_code}")
            
            if validator(resp):
                return True

            # If not 200, but also not a definitive client error (404, 401, 403), retry
            if resp.status_code not in [200, 404, 401, 403] and attempt < max_retries - 1:
                logger.warning(f"{probe_type} model {model_name} returned {resp.status_code}. Retrying...")
                await asyncio.sleep(2 ** attempt)
                continue
            
            break
        except (httpx.ReadTimeout, httpx.ConnectError):
            logger.warning(f"Timeout/Connection error on attempt {attempt + 1} checking {probe_type} model {model_name}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            break
        except Exception as e:
            logger.exception(f"Exception during {probe_type} probe for {model_name}: %s", e)
            break
    return False


async def check_chat_model_ready(model_name: str) -> bool:
    """
    Check if the supplied model is a chat model and is ready in the LLM API/server.
    Returns True if ready, False if not ready or not found.
    Robustly retries on transient error codes.
    """
    cache_key = f"chat:{model_name}"
    cached_status = _check_model_ready_cache(cache_key)
    if cached_status is not None:
        return cached_status

    base_url = _get_base_url()
    try:
        async with httpx.AsyncClient() as client:
            if not await _check_model_exists(client, base_url, model_name):
                return _update_model_ready_cache(cache_key, False)

            def chat_validator(resp: httpx.Response) -> bool:
                if resp.status_code != 200:
                    return False
                try:
                    data = resp.json()
                    
                    # Verify integrity: if the model name in the response same as the requested model name?
                    resp_model = data.get("model", "")
                    if resp_model and model_name not in resp_model and resp_model not in model_name:
                        logger.warning(f"Chat model mismatch! Requested: {model_name}, Got: {resp_model}")
                        return False

                    # Verify it actually generated a message structure
                    choices = data.get("choices", [])
                    return bool(choices and choices[0].get("message", {}).get("content") is not None)
                except Exception:
                    return False

            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1
            }
            
            result = await _probe_with_retry(
                client, 
                f"{base_url}/chat/completions", 
                payload, 
                chat_validator, 
                model_name, 
                "Chat"
            )
            return _update_model_ready_cache(cache_key, result)

    except Exception:
        logging.exception("Exception during chat model readiness check")
        return _update_model_ready_cache(cache_key, False)

async def check_embed_model_ready(model_name: str) -> bool:
    """
    Check if the supplied model is an embedding model and is ready in the LLM API/server.
    Returns True if ready, False if not ready or not found.
    Robustly retries on transient error codes.
    """
    cache_key = f"embed:{model_name}"
    cached_status = _check_model_ready_cache(cache_key)
    if cached_status is not None:
        return cached_status

    base_url = _get_base_url()
    try:
        async with httpx.AsyncClient() as client:
            if not await _check_model_exists(client, base_url, model_name):
                return _update_model_ready_cache(cache_key, False)

            def embed_validator(resp: httpx.Response) -> bool:
                if resp.status_code != 200:
                    return False
                try:
                    data = resp.json()
                    resp_model = data.get("model", "")
                    # Verify integrity: if the model name in the response same as the requested model name?
                    if resp_model and model_name not in resp_model and resp_model not in model_name:
                        logger.warning(f"Embedding model mismatch! Requested: {model_name}, Got: {resp_model}")
                        return False
                    return True
                except Exception:
                    return False

            result = await _probe_with_retry(
                client, 
                f"{base_url}/embeddings", 
                {"model": model_name, "input": "hi"}, 
                embed_validator, 
                model_name, 
                "Embedding"
            )
            return _update_model_ready_cache(cache_key, result)

    except Exception:
        logging.exception("Exception during embedding model readiness check")
        return _update_model_ready_cache(cache_key, False)
