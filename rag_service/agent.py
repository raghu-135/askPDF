import asyncio
import logging
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from models import get_llm, get_embedding_model, get_system_prompt
from vectordb.qdrant import QdrantAdapter

logger = logging.getLogger(__name__)

# Initialize search tool
search_tool = DuckDuckGoSearchRun()


async def invoke_with_retry(func, *args, **kwargs):
    """
    Retry an async function if it fails with a 503 model loading error.
    """
    max_retries = 10
    delay = 5
    for i in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            if "503" in err_str and ("loading" in err_str.lower() or "unavailable" in err_str.lower()):
                logger.warning(f"Model is loading (503). Retrying in {delay}s... (Attempt {i+1}/{max_retries})")
                await asyncio.sleep(delay)
                continue
            raise
    raise Exception("Max retries reached while waiting for model to load.")


async def generate_optimized_search_query(
    question: str,
    context: str,
    history: List[BaseMessage],
    llm_name: str
) -> str:
    """
    Generate an optimized search query using the LLM, considering context and history.
    """
    try:
        llm = get_llm(llm_name)
        
        # Prepare context snippets for search query generation
        history_snippet = ""
        if history:
            history_snippet = "\n".join([f"{getattr(m, 'type', 'user')}: {m.content[:200]}..." for m in history[-3:]])
        
        context_snippet = context[:1500] if context else "No context available."

        query_gen_prompt = (
            "You are a search expert. Generate a simple, plain-text search engine query to find information "
            "that answers the user's question. Use only keywords relevant to the topic.\n\n"
            "CRITICAL: Output ONLY the search query text. Do NOT use JSON, do NOT use quotes, "
            "do NOT explain your reasoning, and do NOT use any specialized search syntax unless "
            "it is a standard Google-style operator.\n\n"
            f"User Question: {question}\n\n"
            f"Recent Conversation:\n{history_snippet}\n\n"
            f"PDF Content (for reference): {context_snippet}\n\n"
            "Search Query:"
        )
        
        response = await invoke_with_retry(llm.ainvoke, [HumanMessage(content=query_gen_prompt)])
        search_query = response.content.strip().strip('"').strip("'")
        
        # Simple heuristic to catch LLM outputting JSON anyway
        if search_query.startswith('{') and '}' in search_query:
            try:
                import json
                data = json.loads(search_query)
                if isinstance(data, dict):
                    search_query = data.get('q') or data.get('query') or data.get('parameters', {}).get('q') or search_query
            except:
                pass

        logger.info(f"Generated search query: {search_query}")
        return search_query
    except Exception as e:
        logger.warning(f"Failed to generate optimized search query: {e}. Using raw question.")
        return question


async def perform_web_search(query: str) -> str:
    """
    Execute a web search and handle common errors/rate limits.
    """
    try:
        # Run search in a separate thread to not block async loop if sync tool
        results = await asyncio.to_thread(search_tool.invoke, query)
        logger.info(f"Web search completed for '{query}'. Results length: {len(results or '')} chars")
        logger.info(f"Web search results: {results}")
        return results
    except Exception as e:
        error_msg = str(e).lower()
        if "ratelimit" in error_msg or "429" in error_msg:
            return "Notice: Web search unavailable due to rate limiting."
        return f"Web search failed: {str(e)}"


class AgentState(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    llm_model: str
    embedding_model: str
    collection_name: Annotated[str, "The name of the Qdrant collection to search"]
    use_web_search: bool
    context: str
    web_context: str
    answer: str
    search_query: str


# Placeholder for a synchronous retrieve function (not implemented)
def retrieve(state: AgentState):
    pass

async def web_search_node(state: AgentState):
    """
    Perform a web search to augment the PDF context.
    Handles rate limiting gracefully.
    """
    question = state["question"]
    context = state.get("context", "")
    history = state.get("chat_history", [])
    llm_name = state["llm_model"]

    search_query = await generate_optimized_search_query(question, context, history, llm_name)
    results = await perform_web_search(search_query)
    
    return {"web_context": results, "search_query": search_query}


async def retrieve_node(state: AgentState):
    """
    Retrieve relevant context from the vector database for the given question.
    """
    question = state["question"]
    emb_model_name = state["embedding_model"]
    embed_model = get_embedding_model(emb_model_name)
    query_vector = await invoke_with_retry(embed_model.aembed_query, question)

    collection_name = state.get("collection_name")
    if not collection_name:
        # Fallback to legacy naming
        base_model_name = emb_model_name.split(":")[0]
        safe_name = base_model_name.replace("-", "_").replace(".", "_").replace("/", "_")
        collection_name = f"rag_{safe_name}"

    db = QdrantAdapter()
    results = await db.search(collection_name, query_vector, limit=5)
    context = "\n\n".join([res["text"] for res in results])
    return {"context": context}

async def generate_node(state: AgentState):
    """
    Generate an answer to the question using the LLM and provided context.
    """
    question = state["question"]
    context = state["context"]
    web_context = state.get("web_context", "")
    llm_name = state["llm_model"]
    llm = get_llm(llm_name)

    # Combine contexts
    full_context = f"PDF Context:\n{context}\n\nWeb Search Context:\n{web_context}"

    system_instruction = get_system_prompt(
        context=full_context,
        use_web=bool(web_context)
    )

    messages = [
        SystemMessage(content=system_instruction),
    ]
    messages.extend(state.get("chat_history", []))
    messages.append(HumanMessage(content=question))

    response = await invoke_with_retry(llm.ainvoke, messages)
    return {"answer": response.content}


# Define the router logic
def router(state: AgentState):
    if state.get("use_web_search", False):
        return "web_search"
    return "generate"

# Define the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")

# Add conditional edge from retrieve
workflow.add_conditional_edges(
    "retrieve",
    router,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()
