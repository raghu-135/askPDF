import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def get_llm(model_name: str, temperature: float = 0.0):
    """
    Returns a configured LLM client.
    Configured for OpenAI-compatible API (DMR).
    """
    base_url = os.getenv("DMR_BASE_URL", "http://host.docker.internal:12434")
    # Helper to clean URL if needed (e.g. ensure /v1 suffix if client requires it, 
    # but langgraph/langchain usually handles base_url + /chat/completions)
    # Typically ChatOpenAI expects base_url to be the root or root/v1
    
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
        api_key="sk-no-key-required" # DMR usually doesn't need a real key
    )

def get_embedding_model(model_name: str):
    """
    Returns a configured Embedding model client.
    """
    base_url = os.getenv("DMR_BASE_URL", "http://host.docker.internal:12434")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    return OpenAIEmbeddings(
        model=model_name,
        base_url=base_url,
        api_key="sk-no-key-required",
        check_embedding_ctx_length=False # Disable check for local models
    )
