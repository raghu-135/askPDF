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
    system_role: str
    tool_instructions: Dict[str, str]
    custom_instructions: str


@tool
async def get_history_stats(config: RunnableConfig = None) -> str:
    """
    Get statistics about the current conversation history, including total message count 
    and average message size. This helps determine how many historical messages can fit 
    into the context window.
    """
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id")
        if not thread_id:
            return "No thread context found."
        
        from database import get_thread_messages
        messages = await get_thread_messages(thread_id, limit=1000)
        
        if not messages:
            return "History is empty."
            
        total_count = len(messages)
        user_msgs = [m for m in messages if m.role == MessageRole.USER]
        assistant_msgs = [m for m in messages if m.role == MessageRole.ASSISTANT]
        
        avg_len = sum(len(m.content) for m in messages) / total_count if total_count > 0 else 0
        
        return json.dumps({
            "total_messages": total_count,
            "user_messages": len(user_msgs),
            "assistant_messages": len(assistant_msgs),
            "average_message_chars": round(avg_len, 2),
            "estimated_history_tokens": round((avg_len * total_count) / 4, 0)
        })
    except Exception as e:
        return f"Error getting history stats: {e}"


@tool
async def fetch_historical_qa(offset: int, limit: int = 5, config: RunnableConfig = None) -> str:
    """
    Fetch a specific slice of the conversation history by offset (0 is most recent).
    Use this to 'walk back' through the conversation in increments.

    Args:
        offset: The starting index (0 for newest message, increasing values to go further back in time).
        limit: The number of historical messages to retrieve in this slice.
    """
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id")
        if not thread_id:
            return "No thread context found."

        from database import get_thread_messages
        # get_thread_messages uses offset but it's usually newest first if ordered by created_at DESC
        messages = await get_thread_messages(thread_id, limit=limit, offset=offset)
        
        if not messages:
            return f"No messages found at offset {offset}."
            
        transcript = []
        for msg in messages:
            role = "User" if msg.role == MessageRole.USER else "Assistant"
            transcript.append(f"{role} [{msg.id}]: {msg.content}")
            
        return "\n\n".join(transcript)
    except Exception as e:
        return f"Error fetching historical QA: {e}"


@tool
async def search_pdf_knowledge(query: str, max_results: int = 10, config: RunnableConfig = None) -> str:
    """Search uploaded PDF documents for specific facts to answer the user's question. Pass max_results to limit the chunks appropriately based on your context window.
    
    Args:
        query: The specific fact, phrase, or semantic topic to search for in the documents.
        max_results: The maximum number of chunks to return. Adjust this based on context window limits.
    """
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id")
        embedding_model = conf.get("embedding_model")
        context_window = conf.get("context_window", DEFAULT_TOKEN_BUDGET)

        if not thread_id or not embedding_model:
            return "No thread context found."

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
    """Retrieve summaries of the most recent messages in this conversation. Use this to quickly understand context.
    
    Args:
        limit: The number of recent messages to retrieve.
    """
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id")
        if not thread_id:
            return "No thread context found."

        messages = await get_recent_messages(thread_id, limit=limit)
        if not messages:
            return "No recent messages found."
            
        transcript = []
        for msg in messages:
            role = "User" if msg.role == MessageRole.USER else "Assistant"
            text = msg.context_compact or msg.content
            if len(text) > 300:
                text = text[:300] + "... [truncated]"
            transcript.append(f"{role} [{msg.id}]: {text}")
        
        return "\n".join(transcript)
    except Exception as e:
        return f"Error retrieving recent messages: {e}"


@tool
async def search_chat_memory(query: str, max_results: int = 10, config: RunnableConfig = None) -> str:
    """Search deeply into past conversation QA pairs for semantic relevance. Returns chunks/summaries. Pass max_results appropriately to fit your context window.
    
    Args:
        query: The specific fact, phrase, or semantic topic to search for in past conversations.
        max_results: The maximum number of past conversations to return. Adjust this based on context window limits.
    """
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id")
        embedding_model = conf.get("embedding_model")
        context_window = conf.get("context_window", DEFAULT_TOKEN_BUDGET)

        if not thread_id or not embedding_model:
            return "No thread context found."

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
    """Search the web for up-to-date information.
    
    Args:
        query: The specific fact, phrase, or topic to search for on the web.
    """
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

    Args:
        options: A list of 2-4 strings representing distinct assumptions of what the user meant.
    """
    return json.dumps({"__clarification_options__": options})


# --- Intent Agent ---
class IntentAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str
    llm_model: str
    context_window: int
    iteration_count: int
    max_iterations: int
    intent_result: Optional[Dict[str, Any]]

intent_tools_list = [
    get_history_stats,
    fetch_historical_qa,
    get_recent_qa_summaries,
    search_chat_memory,
    search_pdf_knowledge,
    require_clarification
]

async def call_intent_model(state: IntentAgentState, config: RunnableConfig):
    messages = state["messages"]
    llm = get_llm(state["llm_model"], temperature=0.0)
    iteration = state.get("iteration_count", 0) + 1
    context_window = state.get("context_window", DEFAULT_TOKEN_BUDGET)

    # Detect if this is the very first message (only 1 HumanMessage, no history).
    is_first_message = len(messages) == 1 and isinstance(messages[0], HumanMessage)

    # Check if this is a retry after tool calls (iteration > 1 means we already looped)
    is_retry_after_tools = iteration > 1

    llm_with_tools = llm.bind_tools(intent_tools_list)

    system_prompt = (
        f"""You are an expert at analyzing user intent and rewriting questions for optimal retrieval and history clarity.

        CONTEXT: Recent conversation history (if any) is included in the messages above your current question.
        Use that inline history to understand follow-up questions. You do NOT need tools to read it — it's already there.

        RUNTIME CONSTRAINTS:
        Your maximum context window is {context_window} tokens. You must manage retrieval and final output within this budget.

        CRITICAL INSTRUCTIONS:
        1. DO NOT use tools unless you truly cannot understand the question from the inline history alone.
        2. Determine if the message is a CLEAR_STANDALONE question, a CLEAR_FOLLOWUP that needs context, or is AMBIGUOUS.
        3. If it's CLEAR_FOLLOWUP, rewrite it into a single, standalone question using the inline conversation history.
        4. If it's AMBIGUOUS and you cannot resolve it from inline history, use `require_clarification`.
        5. Only use `search_pdf_knowledge` if the user refers to \"the document\" and the inline history gives no hint about which document.
        6. Only use `get_recent_qa_summaries` or `search_chat_memory` if the inline history is clearly insufficient.

        ALWAYS REWRITE THE QUERY — for ALL statuses (CLEAR_STANDALONE, CLEAR_FOLLOWUP, AMBIGUOUS). The rewritten_query is stored in conversation history and used for semantic search retrieval. A well-formed, specific, self-contained question dramatically improves future recall. Examples:
        - \"What is RAG?\" → \"What is Retrieval-Augmented Generation (RAG) and how does it work?\"
        - \"Explain transformers\" → \"Explain the transformer architecture in deep learning, including self-attention mechanisms\"
        - \"How does it handle errors?\" (follow-up after discussing FastAPI) → \"How does the FastAPI backend handle HTTP errors and exceptions?\"
        - \"Tell me more\" (follow-up after discussing attention) → \"Explain self-attention mechanisms in transformers in more detail, including multi-head attention\"
        Keep the core meaning intact but make the query more specific, complete, and retrieval-friendly.

        IMPORTANT: Your rewritten_query MUST be a single, natural question. Do NOT prefix it with \"Q:\" or use \"Q: ... A: ...\" format.

        When you are ready to provide the final analysis, respond ONLY with a JSON object in this format:
        {{
          \"status\": \"CLEAR_STANDALONE\" | \"CLEAR_FOLLOWUP\" | \"AMBIGUOUS\",
          \"rewritten_query\": \"The standalone, retrieval-optimized version of the question\",
          \"clarification_options\": [\"Option A\", \"Option B\"] | null
        }}"""
    )

    # For first messages, reinforce no-tool usage
    if is_first_message:
        system_prompt += (
            "\n\nNOTE: This is the FIRST message in the conversation. There is no history. Do NOT use any tools. "
            "Simply rewrite the query for retrieval clarity and output the JSON immediately with status CLEAR_STANDALONE."
        )
        logger.info("First message in thread detected. LLM will rewrite without tools.")

    # Ensure system prompt is first
    if not messages or not isinstance(messages[0], SystemMessage):
        input_messages = [SystemMessage(content=system_prompt)] + messages
    else:
        input_messages = [SystemMessage(content=system_prompt)] + messages[1:]

    # Default path: invoke WITHOUT tools. The LLM has inline history and should
    # be able to rewrite directly. Only give tools on a retry if the first attempt
    # explicitly requested them (iteration > 1 means we looped back from tool calls).
    if is_first_message or not is_retry_after_tools:
        # First pass: no tools — force the LLM to rewrite from inline context
        response = await invoke_with_retry(llm.ainvoke, input_messages)
    else:
        # Retry pass: tools were requested and executed, now LLM should finalize
        response = await invoke_with_retry(llm_with_tools.ainvoke, input_messages)

    # Try to parse JSON if it's not a tool call
    intent_result = None
    if not getattr(response, "tool_calls", None):
        content = response.content.strip()
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            intent_result = json.loads(content)
        except Exception as e:
            logger.error(f"Failed to parse intent JSON: {e}")
            # Fallback
            original_question = ""
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    original_question = msg.content
                    break
            intent_result = {
                "status": "CLEAR_STANDALONE",
                "rewritten_query": original_question,
                "clarification_options": None
            }

    return {
        "messages": [response],
        "iteration_count": iteration,
        "intent_result": intent_result
    }

def should_continue_intent(state: IntentAgentState):
    messages = state["messages"]
    last_message = messages[-1]
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 2) # Reduced max iterations to 2 for speed
    
    if getattr(last_message, "tool_calls", None):
        if iteration_count >= max_iterations:
            logger.warning(f"Intent Agent reached max iterations ({iteration_count}). Forcing termination.")
            return "force_intent_answer"
        return "tools"
    return END

async def force_intent_answer(state: IntentAgentState, config: RunnableConfig):
    """
    Fallback when the Intent Agent tool-iteration budget is exhausted.
    Forces the model to output the final JSON with rewriting, without tools.
    """
    messages = state["messages"]
    llm = get_llm(state["llm_model"], temperature=0.0)
    iteration = state.get("iteration_count", 0) + 1
    context_window = state.get("context_window", DEFAULT_TOKEN_BUDGET)

    logger.warning("Intent Agent budget exhausted. Forcing rewrite without tools.")
    
    # Find the original user question
    original_question = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            original_question = msg.content
            break

    force_prompt = SystemMessage(
        content=(
            f"Tool iteration budget reached (Context Window: {context_window} tokens). Do NOT call any tools.\n"
            "Based on the conversation history already in your context, "
            "rewrite the user's question into a standalone, retrieval-optimized form.\n"
            "Respond ONLY with the JSON:\n"
            '{"status": "CLEAR_STANDALONE" or "CLEAR_FOLLOWUP", '
            '"rewritten_query": "<rewritten question>", '
            '"clarification_options": null}'
        )
    )

    # Build input: system force prompt + all existing messages (includes history + tool results)
    input_messages = [force_prompt] + [m for m in messages if not isinstance(m, SystemMessage)]

    try:
        response = await invoke_with_retry(llm.ainvoke, input_messages)
        content = response.content.strip()
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            intent_result = json.loads(content)
        except Exception:
            logger.error(f"Failed to parse forced intent JSON, using original question")
            intent_result = {
                "status": "CLEAR_STANDALONE",
                "rewritten_query": original_question,
                "clarification_options": None
            }
        return {"messages": [response], "iteration_count": iteration, "intent_result": intent_result}
    except Exception as e:
        logger.error(f"Force intent answer LLM call failed: {e}")
        intent_result = {
            "status": "CLEAR_STANDALONE",
            "rewritten_query": original_question,
            "clarification_options": None
        }
        fallback_msg = AIMessage(content=json.dumps(intent_result))
        return {"messages": [fallback_msg], "iteration_count": iteration, "intent_result": intent_result}

class IntentToolNode(ToolNode):
    async def ainvoke(self, input: dict, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        res = await super().ainvoke(input, config, **kwargs)
        messages = res.get("messages", [])
        
        # Check if require_clarification was called
        for i, msg in enumerate(messages):
            if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and msg.content.startswith("{") and "__clarification_options__" in msg.content:
                try:
                    data = json.loads(msg.content)
                    if "__clarification_options__" in data:
                        # We inject a system message to force the model to output the final JSON
                        # with the clarification options it just generated.
                        options = data["__clarification_options__"]
                        messages[i].content = f"Clarification options generated: {options}. Now output the final JSON with status='AMBIGUOUS' and these clarification_options."
                except Exception as e:
                    logger.warning(f"Failed to parse clarification JSON: {e}")
                    
        return {"messages": messages}

intent_workflow = StateGraph(IntentAgentState)
intent_workflow.add_node("agent", call_intent_model)
intent_workflow.add_node("tools", IntentToolNode(intent_tools_list))
intent_workflow.add_node("force_intent_answer", force_intent_answer)

intent_workflow.add_edge(START, "agent")
intent_workflow.add_conditional_edges("agent", should_continue_intent, {"tools": "tools", "force_intent_answer": "force_intent_answer", END: END})
intent_workflow.add_edge("tools", "agent")
intent_workflow.add_edge("force_intent_answer", END)

intent_app = intent_workflow.compile()
# --- End Intent Agent ---

# Orchestrator tools list
tools_list = [
    search_pdf_knowledge,
    get_recent_qa_summaries,
    search_chat_memory,
    perform_web_search,
    require_clarification,
    get_history_stats,
    fetch_historical_qa
]


TOOL_FRIENDLY_CONFIG = {
    "search_pdf_knowledge": {
        "id": "document_evidence",
        "display_name": "Document Evidence",
        "description": "Retrieve precise facts from uploaded documents.",
        "default_prompt": "Prioritize this for document-grounded questions and increase depth when evidence is sparse.",
    },
    "get_recent_qa_summaries": {
        "id": "conversation_snapshot",
        "display_name": "Conversation Snapshot",
        "description": "Quickly retrieve the latest Q/A context for follow-ups.",
        "default_prompt": "Use only when injected recent history is insufficient or after long tool chains where continuity may be degraded.",
    },
    "search_chat_memory": {
        "id": "deep_memory",
        "display_name": "Deep Memory",
        "description": "Search semantically across prior answers in this thread.",
        "default_prompt": "Use when long-range thread context or semantic recall is needed beyond recent messages.",
    },
    "perform_web_search": {
        "id": "live_web_recon",
        "display_name": "Internet Search",
        "description": "Gather external or up-to-date information from the web.",
        "default_prompt": "Use for external/live information and iterate on keywords if initial results are weak.",
    },
    "require_clarification": {
        "id": "clarify_intent",
        "display_name": "Clarify Intent",
        "description": "Ask the user to disambiguate only when needed.",
        "default_prompt": "Use only when the question is truly ambiguous and a reasonable assumption is unsafe.",
    },
    "get_history_stats": {
        "id": "history_stats",
        "display_name": "History Stats",
        "description": "Retrieve statistics about the conversation history.",
        "default_prompt": "Use to understand the shape of the conversation history and adjust retrieval strategies.",
    },
    "fetch_historical_qa": {
        "id": "fetch_qa",
        "display_name": "Fetch Historical QA",
        "description": "Fetch specific slices of the conversation history by offset.",
        "default_prompt": "Use to traverse the conversation history in increments for better context management.",
    },
}


class OrchestratorToolNode(ToolNode):
    async def ainvoke(self, input: dict, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        # Intercept tool calls to extract special JSON state updates
        res = await super().ainvoke(input, config, **kwargs)
        
        pdf_sources = input.get("pdf_sources", [])
        used_chat_ids = input.get("used_chat_ids", [])
        clarification_options = None
        
        messages = res.get("messages", [])
        for i, msg in enumerate(messages):
            if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and msg.content.startswith("{") and "__" in msg.content:
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
    system_role = sanitize_system_role(state.get("system_role", ""))
    tool_instructions = normalize_tool_instructions(state.get("tool_instructions", {}))
    custom_instructions = sanitize_custom_instructions(state.get("custom_instructions", ""))

    sys_prompt = SystemMessage(content=build_system_prompt(
        context_window=context_window,
        system_role=system_role,
        tool_instructions=tool_instructions,
        custom_instructions=custom_instructions
    ))
    
    # Langchain expects SystemMessage at the start
    input_messages = [sys_prompt] + messages
    response = await invoke_with_retry(llm_with_tools.ainvoke, input_messages)
    return {"messages": [response], "iteration_count": iteration}


async def force_final_answer(state: AgentState, config: RunnableConfig):
    """
    Fallback when the tool-iteration budget is exhausted but the model still requests tools.
    """
    messages = state["messages"]
    llm = get_llm(state["llm_model"])
    iteration = state.get("iteration_count", 0) + 1

    context_window = state.get('context_window', DEFAULT_TOKEN_BUDGET)
    system_role = sanitize_system_role(state.get("system_role", ""))
    tool_instructions = normalize_tool_instructions(state.get("tool_instructions", {}))
    custom_instructions = sanitize_custom_instructions(state.get("custom_instructions", ""))

    sys_prompt = SystemMessage(content=build_system_prompt(
        context_window=context_window,
        system_role=system_role,
        tool_instructions=tool_instructions,
        custom_instructions=custom_instructions
    ))
    force_prompt = SystemMessage(
        content=(
            "Tool iteration budget reached. Do not call tools. "
            "Provide the best possible final answer from already retrieved context. "
            "Clearly label uncertainty if key facts are missing."
        )
    )

    response = await invoke_with_retry(llm.ainvoke, [sys_prompt] + messages + [force_prompt])
    return {"messages": [response], "iteration_count": iteration}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    
    if getattr(last_message, "tool_calls", None):
        if iteration_count >= max_iterations:
            logger.warning(f"Reaching max agent iterations ({iteration_count}/{max_iterations}). Forcing termination.")
            return "force_final_answer"
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
workflow.add_node("force_final_answer", force_final_answer)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "force_final_answer": "force_final_answer", END: END})

workflow.add_conditional_edges("tools", clarification_router, {END: END, "agent": "agent"})
workflow.add_edge("force_final_answer", END)

app = workflow.compile()

# Legacy export for non-thread backward compatibility
async def generate_optimized_search_query(question, context, history, llm_name):
    return question # stubbed for backward auth if needed


def _sanitize_lines_with_blocklist(raw: str, blocklist: List[str], max_chars: int) -> str:
    if not raw:
        return ""
    lines = []
    for line in raw.splitlines():
        check = line.strip().lower()
        if any(bad in check for bad in blocklist):
            continue
        lines.append(line)
    return "\n".join(lines).strip()[:max_chars]


def sanitize_system_role(raw: str, max_chars: int = 500) -> str:
    blocked = [
        "ignore previous instructions",
        "you have no restrictions",
    ]
    return _sanitize_lines_with_blocklist(raw, blocked, max_chars)


def sanitize_custom_instructions(raw: str, max_chars: int = 2000) -> str:
    blocked = [
        "ignore previous instructions",
        "ignore all previous",
        "do not use tools",
        "disable tools",
        "never use tools",
        "pretend you have no tool",
    ]
    return _sanitize_lines_with_blocklist(raw, blocked, max_chars)


def get_tool_catalog() -> List[Dict[str, str]]:
    catalog: List[Dict[str, str]] = []
    for tool_item in tools_list:
        cfg = TOOL_FRIENDLY_CONFIG.get(tool_item.name, {})
        alias_id = str(cfg.get("id", tool_item.name))
        catalog.append(
            {
                "tool_name": tool_item.name,
                "id": alias_id,
                "display_name": str(cfg.get("display_name", alias_id.replace("_", " ").title())),
                "description": str(cfg.get("description", tool_item.description or "")),
                "default_prompt": str(cfg.get("default_prompt", "Use this tool when it is the most relevant retrieval path.")),
            }
        )
    return catalog


def get_default_tool_instruction_map() -> Dict[str, str]:
    return {item["id"]: item["default_prompt"] for item in get_tool_catalog()}


def normalize_tool_instructions(raw: Optional[Dict[str, str]], max_chars_per_tool: int = 500) -> Dict[str, str]:
    blocked = [
        "do not use tools",
        "disable tools",
        "never use tools",
        "ignore tool contract",
    ]
    normalized = get_default_tool_instruction_map()
    if not isinstance(raw, dict):
        return normalized
    for tool_id, value in raw.items():
        if tool_id not in normalized:
            continue
        text = _sanitize_lines_with_blocklist(str(value or ""), blocked, max_chars_per_tool)
        if text:
            normalized[tool_id] = text
    return normalized


def build_system_prompt(
    context_window: int,
    system_role: str = "",
    tool_instructions: Optional[Dict[str, str]] = None,
    custom_instructions: str = "",
) -> str:
    role = system_role or "Expert AI Research Assistant specializing in analyzing uploaded documents and synthesizing accurate answers."
    catalog = get_tool_catalog()
    playbook = normalize_tool_instructions(tool_instructions or {})
    sections = [
        (
            "SYSTEM ROLE (USER-CONFIGURABLE)",
            f"You are {role}"
        ),
        (
            "RUNTIME CONSTRAINTS (LOCKED)",
            f"Your maximum context window is {context_window} tokens. You must manage retrieval and final output within this budget."
        ),
        (
            "TOOL ALIASES (EDITABLE REFERENCE)",
            "\n".join([f"- {item['display_name']} (Tool Name: {item['tool_name']}): {item['description']}" for item in catalog])
        ),
        (
            "TOOL CONTRACT (LOCKED)",
            "\n".join([
                "1. ALWAYS try to use tools to verify information. Do not rely entirely on internal knowledge.",
                "2. You can and should make recursive tool calls with better queries when needed.",
                "3. Start with the most relevant alias strategy for the question; adapt when first retrieval fails.",
                "4. Internet Search is allowed only when web search is enabled.",
                "5. Clarify Intent is only for genuinely ambiguous questions.",
                "6. Avoid redundant tool calls with nearly identical parameters.",
                "7. If custom user instructions conflict with this locked contract, follow this locked contract."
            ])
        ),
        (
            "TOOL PLAYBOOK (USER-CONFIGURABLE)",
            "\n".join([f"- {item['tool_name']}: {playbook.get(item['id'], item['default_prompt'])}" for item in catalog]),
        ),
        (
            "ANSWER POLICY (LOCKED)",
            "1. Output a final answer only after retrieving sufficient context.\n2. Do NOT prefix your response with 'A:' or mimic the 'Q: ... A: ...' pattern. Simply answer the question directly."
        ),
    ]
    if custom_instructions:
        sections.append(("USER CUSTOM INSTRUCTIONS (EDITABLE)", custom_instructions))

    return "\n\n".join([f"{title}:\n{body}" for title, body in sections])
