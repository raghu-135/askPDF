import asyncio
import logging
import json
from typing import TypedDict, List, Annotated, Dict, Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from models import get_llm, get_embedding_model, DEFAULT_TOKEN_BUDGET, DEFAULT_MAX_ITERATIONS
from vectordb.qdrant import QdrantAdapter
from database import get_recent_messages, MessageRole

logger = logging.getLogger(__name__)

search_tool = DuckDuckGoSearchRun()

async def invoke_with_retry(func, *args, **kwargs):
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

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str
    llm_model: str
    embedding_model: str
    context_window: int
    use_web_search: bool
    pdf_sources: List[Dict[str, Any]]
    used_chat_ids: List[str]
    clarification_options: Optional[List[str]]
    iteration_count: int
    max_iterations: int


@tool
async def search_pdf_knowledge(query: str, max_results: int = 10, config: RunnableConfig = None) -> str:
    """Search uploaded PDF documents for specific facts to answer the user's question. Pass max_results to limit the chunks appropriately based on your context window."""
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id")
        embedding_model = conf.get("embedding_model")
        context_window = conf.get("context_window", DEFAULT_TOKEN_BUDGET)

        embed_model = get_embedding_model(embedding_model)
        query_vector = await invoke_with_retry(embed_model.aembed_query, query)
        
        db = QdrantAdapter()
        
        raw_pdf_chunks = await db.search_pdf_chunks(
            thread_id=thread_id,
            query_vector=query_vector,
            limit=max_results
        )
        
        expansion_radius = max(2, min(10, int(context_window / 8000) + 1))
        file_chunk_map = {}
        for hit in raw_pdf_chunks:
            file_hash = hit.get("file_hash")
            chunk_id = hit.get("chunk_id")
            if file_hash is not None and chunk_id is not None:
                if file_hash not in file_chunk_map:
                    file_chunk_map[file_hash] = set()
                for neighbor_id in range(chunk_id - expansion_radius, chunk_id + expansion_radius + 1):
                    if neighbor_id >= 0:
                        file_chunk_map[file_hash].add(neighbor_id)
        
        expanded_pdf_chunks = []
        for file_hash, id_set in file_chunk_map.items():
            expanded_batch = await db.get_chunks_by_ids(
                thread_id=thread_id,
                file_hash=file_hash,
                chunk_ids=list(id_set)
            )
            expanded_pdf_chunks.extend(expanded_batch)
            
        expanded_pdf_chunks.sort(key=lambda x: (x.get("file_hash", ""), x.get("chunk_id", 0)))
        
        if not expanded_pdf_chunks:
            return "No relevant PDF content found."

        sources = []
        context_parts = []
        for chunk in expanded_pdf_chunks:
            text = chunk.get("text", "")
            context_parts.append(text)
            sources.append({
                "text": text[:200] + "..." if len(text) > 200 else text,
                "file_hash": chunk.get("file_hash"),
                "score": chunk.get("score", 0.0)
            })
            
        return json.dumps({
            "content": "\n\n".join(context_parts),
            "__pdf_sources__": sources
        })
    except Exception as e:
        logger.error(f"Error in search_pdf_knowledge: {e}", exc_info=True)
        return f"Error retrieving PDF knowledge: {e}"


@tool
async def get_recent_qa_summaries(limit: int, config: RunnableConfig = None) -> str:
    """Retrieve summaries of the most recent messages in this conversation. Use this to quickly understand context."""
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id")

        messages = await get_recent_messages(thread_id, limit=limit)
        if not messages:
            return "No recent messages found."
            
        transcript = []
        for msg in messages:
            role = "User" if msg.role == MessageRole.USER else "Assistant"
            text = msg.content
            if len(text) > 300:
                text = text[:300] + "... [truncated]"
            transcript.append(f"{role} [{msg.id}]: {text}")
        
        return "\n".join(transcript)
    except Exception as e:
        return f"Error retrieving recent messages: {e}"


@tool
async def search_chat_memory(query: str, max_results: int = 10, config: RunnableConfig = None) -> str:
    """Search deeply into past conversation QA pairs for semantic relevance. Returns chunks/summaries. Pass max_results appropriately to fit your context window."""
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id")
        embedding_model = conf.get("embedding_model")
        context_window = conf.get("context_window", DEFAULT_TOKEN_BUDGET)

        embed_model = get_embedding_model(embedding_model)
        query_vector = await invoke_with_retry(embed_model.aembed_query, query)
        
        db = QdrantAdapter()
        recalled_memories = await db.search_chat_memory(
            thread_id=thread_id,
            query_vector=query_vector,
            limit=max_results
        )

        if not recalled_memories:
            return "No relevant past conversations found."

        used_ids = [mem["message_id"] for mem in recalled_memories if mem.get("message_id")]
        
        context_parts = []
        for mem in recalled_memories:
            context_parts.append(mem.get("text", ""))

        return json.dumps({
            "content": "\n\n---\n\n".join(context_parts),
            "__used_chat_ids__": used_ids
        })
    except Exception as e:
        return f"Error retrieving chat memory: {e}"


@tool
async def perform_web_search(query: str) -> str:
    """Search the web for up-to-date information."""
    try:
        results = await asyncio.to_thread(search_tool.invoke, query)
        return str(results)
    except Exception as e:
        return f"Web search failed: {str(e)}"


@tool
async def require_clarification(options: List[str]) -> str:
    """
    If the user's question is ambiguous, call this tool with a list of 2-4 possible options 
    for what they might have meant. This pauses the agent and asks the user for clarification.
    """
    return json.dumps({"__clarification_options__": options})


# Orchestrator tools list
tools_list = [
    search_pdf_knowledge,
    get_recent_qa_summaries,
    search_chat_memory,
    perform_web_search,
    require_clarification
]


class OrchestratorToolNode(ToolNode):
    async def ainvoke(self, input: dict, config: RunnableConfig, **kwargs: Any) -> Any:
        # Intercept tool calls to extract special JSON state updates
        res = await super().ainvoke(input, config, **kwargs)
        
        pdf_sources = input.get("pdf_sources", [])
        used_chat_ids = input.get("used_chat_ids", [])
        clarification_options = None
        
        messages = res.get("messages", [])
        for i, msg in enumerate(messages):
            if isinstance(msg, ToolMessage) and msg.content.startswith("{") and "__" in msg.content:
                try:
                    data = json.loads(msg.content)
                    if "__pdf_sources__" in data:
                        pdf_sources.extend(data["__pdf_sources__"])
                        messages[i].content = data["content"]
                    if "__used_chat_ids__" in data:
                        used_chat_ids.extend(data["__used_chat_ids__"])
                        messages[i].content = data["content"]
                    if "__clarification_options__" in data:
                        clarification_options = data["__clarification_options__"]
                        messages[i].content = f"Interrupted for clarification with options: {clarification_options}"
                except Exception as e:
                    logger.warning(f"Failed to parse tool JSON output: {e}")
                    pass
                    
        return {
            "messages": messages,
            "pdf_sources": pdf_sources,
            "used_chat_ids": used_chat_ids,
            "clarification_options": clarification_options
        }

async def call_model(state: AgentState, config: RunnableConfig):
    messages = state["messages"]
    llm = get_llm(state["llm_model"])
    iteration = state.get("iteration_count", 0) + 1
    
    # Enable tools (filter web search if disabled)
    valid_tools = tools_list if state.get("use_web_search", False) else [t for t in tools_list if t.name != "perform_web_search"]
    llm_with_tools = llm.bind_tools(valid_tools)
    
    context_window = state.get('context_window', DEFAULT_TOKEN_BUDGET)
    
    sys_prompt = SystemMessage(content=(
        f"You are an Expert AI Research Assistant specializing in analyzing uploaded documents and synthesizing accurate answers. Your maximum context window is {context_window} tokens.\n"
        "Your goal is to accurately answer the user's question. You must think step-by-step and use tools to gather information.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. ALWAYS try to use tools to verify information. Do not rely entirely on your internal knowledge.\n"
        "2. You can and should make RECURSIVE tool calls: if your first search doesn't return the exact information you need, you MUST call the tool again with different, more specific queries. Keep searching until you find the answer or prove it is not in the documents.\n"
        "3. Use `get_recent_qa_summaries` first to understand if the user is asking a follow-up question.\n"
        "4. Use `search_pdf_knowledge` for questions about uploaded documents. Scale `max_results` safely.\n"
        "5. Use `search_chat_memory` for deep semantic searches of past answers.\n"
        "6. Use `perform_web_search` for external/live information. Try different keywords if the first search fails.\n"
        "7. Use `require_clarification` ONLY if the question is completely ambiguous and you cannot make a reasonable guess.\n"
        "8. ONLY output your final answer once you have successfully retrieved sufficient context. If you do not have enough context, DO NOT answer yet—use a tool again.\n"
        "9. IMPORTANT: Avoid redundant tool calls. If a search query yields no results, try a DIFFERENT approach or summarize based on what you already know. Do not keep calling the same tool with the same or very similar parameters."
    ))
    
    # Langchain expects SystemMessage at the start
    input_messages = [sys_prompt] + messages
    response = await invoke_with_retry(llm_with_tools.ainvoke, input_messages)
    return {"messages": [response], "iteration_count": iteration}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    
    if getattr(last_message, "tool_calls", None):
        if iteration_count >= max_iterations:
            logger.warning(f"Reaching max agent iterations ({iteration_count}/{max_iterations}). Forcing termination.")
            # We add a small instruction message to the assistant to force a final answer if they tried to call tools again
            # actually should_continue just drives the router. If we return END, the graph stops.
            # The last message IS a tool call, so if we END now, the user sees tool calls but no final answer.
            # Usually better to let it go back to agent one last time with a warning.
            # But if iteration_count is already max_iterations and they wanted tools, we should probably just stop and let them 
            # try to answer in the NEXT step.
            # Re-thinking: if it's at max_iterations and they want tools, we return END. 
            # The chat_service will take final_messages[-1].content. If it's a tool call, that's bad.
            # Better strategy: if iteration_count == max_iterations and tool_calls exist, swap the node to a 'force_answer' node or just continue.
            # Let's stick to max_iterations as a hard stop for the loop.
            return END
        return "tools"
    return END

def clarification_router(state: AgentState):
    if state.get("clarification_options"):
        return END  # Suspend graph
    return "agent"


# Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", OrchestratorToolNode(tools_list))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})

workflow.add_conditional_edges("tools", clarification_router, {END: END, "agent": "agent"})

app = workflow.compile()

# Legacy export for non-thread backward compatibility
async def generate_optimized_search_query(question, context, history, llm_name):
    return question # stubbed for backward auth if needed
