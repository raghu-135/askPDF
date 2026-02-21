import asyncio
import logging
from typing import TypedDict, List, Annotated, Dict, Any, Union
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END, START, add_messages
from langgraph.prebuilt import ToolNode
from models import get_llm, get_embedding_model, get_system_prompt, DEFAULT_TOKEN_BUDGET
from vectordb.qdrant import QdrantAdapter
from langchain_core.runnables import RunnableConfig
from database import get_recent_messages

logger = logging.getLogger(__name__)

# Try to use tiktoken for token counting if available, else use a fallback.
try:
    import tiktoken
    ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODER = None

def count_tokens(text: str) -> int:
    """Accurately count tokens if tiktoken is available, else estimate."""
    if ENCODER is not None:
        return len(ENCODER.encode(text))
    # Fallback to rough estimate: 4 chars per token
    return len(text) // 4

async def trim_relevant_messages(
    messages: List[BaseMessage],
    max_tokens: int,
    embedding_model_name: str
) -> List[BaseMessage]:
    """
    Intelligently prunes history based on semantic relevance to the active question.
    Preserves System Message, Active Turn, and most relevant historical interactions.
    """
    if not messages:
        return []
        
    # Check total tokens first. If within budget, return as is.
    total_tokens = sum(count_tokens(m.content) for m in messages)
    if total_tokens <= max_tokens:
        return messages

    logger.info(f"--- [Context Intelligence] Pruning needed: {total_tokens} > {max_tokens}. Identifying relevant messages... ---")

    # 1. System Message (Always preserve)
    system_msg = None
    other_msgs = messages[:]
    if isinstance(messages[0], SystemMessage):
        system_msg = messages[0]
        other_msgs = messages[1:]

    # 2. Identify the Active Turn (Last Human message onwards)
    last_human_idx = -1
    for i in range(len(other_msgs) - 1, -1, -1):
        if isinstance(other_msgs[i], HumanMessage):
            last_human_idx = i
            break
            
    if last_human_idx == -1:
        # Fallback if no human message found (just return common tail)
        return ([system_msg] + other_msgs[-10:]) if system_msg else other_msgs[-10:]

    history_msgs = other_msgs[:last_human_idx]
    active_turn = other_msgs[last_human_idx:]
    active_query = active_turn[0].content

    # 3. Group History into Dialogue Turns
    turns = []
    current_turn_msgs = []
    for msg in history_msgs:
        if isinstance(msg, HumanMessage):
            if current_turn_msgs:
                turns.append(current_turn_msgs)
            current_turn_msgs = [msg]
        else:
            current_turn_msgs.append(msg)
    if current_turn_msgs:
        turns.append(current_turn_msgs)

    if not turns:
        return ([system_msg] + active_turn) if system_msg else active_turn

    # 4. Semantic Relevance Selection
    try:
        embed_model = get_embedding_model(embedding_model_name)
        turn_queries = [t[0].content for t in turns]
        all_vecs = await embed_model.aembed_documents([active_query] + turn_queries)
        active_vec = all_vecs[0]
        historical_vecs = all_vecs[1:]
        
        # Calculate cosine similarities (manual dot product)
        turn_relevance = []
        for idx, vec in enumerate(historical_vecs):
            score = sum(a*b for a, b in zip(active_vec, vec))
            turn_relevance.append((score, idx))
    except Exception as e:
        logger.error(f"Semantic scoring failed: {e}. Falling back to recent history.")
        # Fallback: Treat all history as potentially relevant and trim from back only
        turn_relevance = [(0.0, i) for i in range(len(turns))]

    # 5. Determine which turns to keep
    kept_indices = set()
    
    # Priority A: Most recent 2 turns
    recent_turns = list(range(max(0, len(turns) - 2), len(turns)))
    for idx in recent_turns:
        kept_indices.add(idx)
        
    # Priority B: Highest relevance scores that fit in remaining budget
    sys_tokens = count_tokens(system_msg.content if system_msg else "")
    active_tokens = sum(count_tokens(m.content) for m in active_turn)
    # Plus tokens from recent turns already selected
    recent_tokens = sum(sum(count_tokens(m.content) for m in turns[i]) for i in recent_turns)
    
    budget_remains = max_tokens - sys_tokens - active_tokens - recent_tokens - 150 # Buffer
    
    # Sort history turns by similarity
    for score, original_idx in sorted(turn_relevance, key=lambda x: x[0], reverse=True):
        if original_idx in kept_indices:
            continue
            
        turn_tokens = sum(count_tokens(m.content) for m in turns[original_idx])
        if turn_tokens < budget_remains and score > 0.4: # Only keep relatively relevant ones
            kept_indices.add(original_idx)
            budget_remains -= turn_tokens

    # 6. Re-assemble in chronological order
    final_messages = []
    if system_msg: final_messages.append(system_msg)
    
    for i in range(len(turns)):
        if i in kept_indices:
            final_messages.extend(turns[i])
            
    final_messages.extend(active_turn)
    
    logger.info(f"--- [Context Intelligence] Using {len(kept_indices)} out of {len(turns)} historical turns. ---")
    return final_messages

async def invoke_with_retry(func, *args, **kwargs):
    """Utility to invoke a function with retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(1 * (attempt + 1))

# Initialize search tool
ddg_search = DuckDuckGoSearchRun()

@tool
async def web_search(query: str) -> str:
    """Useful when you need to search the internet for information that is not in the PDF."""
    logger.info(f"--- [Tool: web_search] query='{query}' ---")
    try:
        results = await asyncio.to_thread(ddg_search.invoke, query)
        content_len = len(results or "")
        logger.info(f"--- [Tool: web_search] Completed. Web Results : {results[:1000]} ---")
        return results
    except Exception as e:
        logger.error(f"--- [Tool: web_search] Failed: {str(e)} ---")
        error_msg = str(e).lower()
        if "ratelimit" in error_msg or "429" in error_msg:
            return "Notice: Web search unavailable due to rate limiting."
        return f"Web search failed: {str(e)}"

@tool
async def search_pdf(query: str, config: RunnableConfig, limit: int = 5) -> str:
    """Useful when you need to search the uploaded PDF for content. 
    Use this to find specific technical details, facts, or descriptions from the documents provided.
    You can specify a 'limit' (default 5, max 20) to retrieve more or fewer results.
    If the initial search doesn't give you what you need, call it again with a different query or higher limit.
    """
    logger.info(f"--- [Tool: search_pdf] query='{query}', limit={limit} ---")
    thread_id = config.get("configurable", {}).get("thread_id")
    emb_model_name = config.get("configurable", {}).get("embedding_model")
    
    if not thread_id:
        logger.warning("--- [Tool: search_pdf] Missing thread_id ---")
        return "Error: thread_id not provided in config."
    
    db = QdrantAdapter()
    embed_model = get_embedding_model(emb_model_name)
    query_vector = await invoke_with_retry(embed_model.aembed_query, query)
    
    if not query_vector:
        logger.error("--- [Tool: search_pdf] Embedding failed ---")
        return "Error: Failed to embed the query."
    
    # 1. Primary search
    safe_limit = min(max(1, limit), 20)
    hits = await db.search_pdf_chunks(thread_id, query_vector, limit=safe_limit)
    logger.info(f"--- [Tool: search_pdf] Found {len(hits)} hits ---")
    
    # 2. Context expansion
    file_chunk_map = {}
    for hit in hits:
        f_hash = hit.get("file_hash")
        c_id = hit.get("chunk_id")
        if f_hash and c_id is not None:
            if f_hash not in file_chunk_map:
                file_chunk_map[f_hash] = set()
            for i in range(c_id - 2, c_id + 3):
                if i >= 0:
                    file_chunk_map[f_hash].add(i)
    
    expanded_chunks = []
    for f_hash, ids in file_chunk_map.items():
        batch = await db.get_chunks_by_ids(thread_id, f_hash, list(ids))
        expanded_chunks.extend(batch)
    
    expanded_chunks.sort(key=lambda x: (x.get("file_hash", ""), x.get("chunk_id", 0)))
    logger.info(f"--- [Tool: search_pdf] Expanded to {len(expanded_chunks)} total segments ---")
    
    context = ""
    for chunk in expanded_chunks:
        context += f"\n---\nSource PDF Content (File: {chunk.get('file_hash', 'N/A')[:8]}): \n{chunk['text']}"
    
    logger.info(f"--- [Tool: search_pdf] PDF search completed. Results : ({context[:1000]}) ---")
    return context if context else "No relevant information found in the PDF."

@tool
async def search_chat_memory(query: str, config: RunnableConfig, limit: int = 5) -> str:
    """Useful when you need to recall past interactions or information shared in the current conversation thread.
    You can specify a 'limit' (default 5, max 15) if you want to see more previous messages.
    """
    logger.info(f"--- [Tool: search_chat_memory] query='{query}', limit={limit} ---")
    thread_id = config.get("configurable", {}).get("thread_id")
    emb_model_name = config.get("configurable", {}).get("embedding_model")
    
    if not thread_id:
        logger.warning("--- [Tool: search_chat_memory] Missing thread_id ---")
        return "Error: thread_id not provided in config."
        
    db = QdrantAdapter()
    embed_model = get_embedding_model(emb_model_name)
    query_vector = await invoke_with_retry(embed_model.aembed_query, query)
    
    if not query_vector:
        logger.error("--- [Tool: search_chat_memory] Embedding failed ---")
        return "Error: Failed to embed the query."
        
    safe_limit = min(max(1, limit), 15)
    results = await db.search_chat_memory(thread_id, query_vector, limit=safe_limit)
    logger.info(f"--- [Tool: search_chat_memory] Recalled {len(results)} past interactions ---")
    
    if not results:
        return "No relevant past conversations found."
        
    memory_context = "\n".join([f"Previous Interaction (Score: {res.get('score', 0):.2f}):\n{res['text']}" for res in results])
    logger.info(f"--- [Tool: search_chat_memory] Memory search completed. Results : ({memory_context[:1000]}) ---")
    return memory_context

@tool
async def get_last_messages(limit: int, config: RunnableConfig) -> str:
    """Useful when you need to see the exact recent history of this conversation in chronological order.
    Use this if the immediate context window in your prompt is too small and you need to see what was just discussed.
    'limit' specifies how many recent messages to retrieve (e.g., 5 or 10).
    """
    logger.info(f"--- [Tool: get_last_messages] limit={limit} ---")
    thread_id = config.get("configurable", {}).get("thread_id")
    
    if not thread_id:
        logger.warning("--- [Tool: get_last_messages] Missing thread_id ---")
        return "Error: thread_id not provided in config."
        
    messages = await get_recent_messages(thread_id, limit=limit)
    logger.info(f"--- [Tool: get_last_messages] Retrieved {len(messages)} messages ---")
    
    if not messages:
        return "No message history found for this thread."
        
    history = "\n".join([f"[{m.role.upper()}]: {m.content}" for m in messages])
    return f"Recent Conversation History (last {limit} messages):\n{history}"

# --- New Logic ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    llm_model: str
    embedding_model: str
    thread_id: str

async def call_model(state: AgentState, config: RunnableConfig):
    """
    Main node of the agent: Decides which tool to call or generates the final answer.
    """
    llm_name = state["llm_model"]
    messages = state["messages"]
    logger.info(f"--- [Node: agent] Starting cycle with {len(messages)} messages ---")
    
    # We don't use get_system_prompt here as the tool outputs will be in the message history
    # Instead we inject a base system prompt as the first message if not present
    if not any(isinstance(m, SystemMessage) for m in messages):
        system_msg = SystemMessage(content=(
            "You are a highly capable AI assistant with access to these specialized tools:\n"
            "1. get_last_messages: Retrieve the most recent messages in chronological order (linear history).\n"
            "2. search_chat_memory: Recall earlier parts of this conversation via vector search (semantic memory).\n"
            "3. search_pdf: Search the uploaded PDF for content.\n"
            "4. web_search: Search the internet for general knowledge.\n\n"
            "STRATEGY:\n"
            "- If user refers to 'it' or 'previous thing' and it isn't in your immediate prompt, try 'get_last_messages' first.\n"
            "- Use 'search_chat_memory' for keywords or concepts from much earlier in the session.\n"
            "- Use 'search_pdf' for document-specific queries.\n"
            "- BE RECURSIVE: If one tool doesn't give you what you need, try another tool or refine your query."
        ))
        messages = [system_msg] + messages

    llm = get_llm(llm_name)
    tools = [search_pdf, search_chat_memory, web_search, get_last_messages]
    llm_with_tools = llm.bind_tools(tools)
    
    # Context window management
    # Ensure current state fits within the model's window before invoking.
    token_budget = DEFAULT_TOKEN_BUDGET
    messages = await trim_relevant_messages(
        messages=messages, 
        max_tokens=token_budget, 
        embedding_model_name=state["embedding_model"]
    )
    
    response = await invoke_with_retry(llm_with_tools.ainvoke, messages, config=config)
    
    tool_calls = getattr(response, "tool_calls", [])
    if tool_calls:
        tool_names = [tc["name"] for tc in tool_calls]
        logger.info(f"--- [Node: agent] LLM requesting tool(s): {tool_names} ---")
    else:
        logger.info(f"--- [Node: agent] LLM generated final answer ---")

    return {"messages": [response]}

# Define the workflow graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode([search_pdf, search_chat_memory, web_search, get_last_messages]))

# Set entry point
workflow.set_entry_point("agent")

# Add edges
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes tool calls, we continue to the tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, we stop
    return "end"

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

workflow.add_edge("tools", "agent")

# Compile the workflow
app = workflow.compile()
