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
    pre_fetch_bundle: Optional[Dict[str, Any]]


@tool
async def get_conversation_stats(config: RunnableConfig = None) -> str:
    """
    A lightweight meta-tool that returns statistics about the current conversation:
    message count, user/assistant split, average message length, and estimated token usage.

    Use this as a PLANNING step before calling `page_conversation_history` to decide appropriate
    offset and limit values. This tool returns no message content — do not use it when
    `search_conversation_history` (semantic search) can answer the question.
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
async def page_conversation_history(offset: int, limit: int = 5, config: RunnableConfig = None) -> str:
    """
    Retrieve a time-ordered slice of verbatim conversation messages by position.
    Messages are ordered newest-first (offset=0 returns the most recent messages).
    Increment offset by `limit` on each call to page sequentially through the full history.

    Prefer `search_conversation_history` for semantic or thematic recall. Use this tool when you
    need to read the exact verbatim text of messages at a known chronological position,
    or when paging through the full conversation in order. Call `get_conversation_stats` first
    if you need to know the total message count before choosing an offset.

    Args:
        offset: Starting position in the message list (0 = most recent). Increase by
                `limit` on each subsequent call to walk further back through history.
        limit: Number of messages to return in this slice (default: 5).
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
async def search_documents(query: str, max_results: int = 10, config: RunnableConfig = None) -> str:
    """
    Perform a semantic vector search over all uploaded PDF documents for this thread.
    Returns the most relevant chunks along with neighboring context passages for continuity.

    This uses embedding-based similarity — phrase queries as natural questions rather than
    keyword strings for best results. If the first call returns weak or irrelevant evidence,
    retry with a rephrased or more specific query before concluding the information is absent.

    Args:
        query: A natural-language question or description of the fact to locate in the PDFs.
        max_results: Number of seed chunks to retrieve before context expansion. Increase
                     for broad topics; decrease when the context window is nearly full.
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
        logger.error(f"Error in search_documents: {e}", exc_info=True)
        return f"Error retrieving PDF knowledge: {e}"


@tool
async def get_recent_context(limit: int, config: RunnableConfig = None) -> str:
    """
    Retrieve the N most recent conversation turns (user + assistant), ordered newest-first.
    Returns compact summaries when available, otherwise truncated full message content.

    Use this for lightweight recency context — for example, to understand what was just
    discussed before the current question. For semantic or thematic recall across a longer
    history, prefer `search_conversation_history` instead.

    Args:
        limit: Number of recent messages to retrieve.
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
async def search_conversation_history(query: str, max_results: int = 10, config: RunnableConfig = None) -> str:
    """
    Perform a semantic vector search across all past conversation QA pairs in this thread.
    Returns the most relevant past exchanges ranked by embedding similarity, regardless
    of when they occurred in the conversation.

    Use this for thematic or conceptual recall (e.g., "what did we previously discuss
    about X?"). Unlike `page_conversation_history`, which returns verbatim messages by time
    position, this finds semantically related content from anywhere in the history.
    Retry with a rephrased query if initial results are off-topic.

    Args:
        query: A natural-language description of the topic or fact to recall from prior exchanges.
        max_results: Maximum number of past QA pairs to return.
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
async def search_web(query: str) -> str:
    """
    Search the web for external, real-time, or post-training knowledge that is not
    available in the uploaded documents or conversation history.

    Only use this when: (a) the question explicitly requires current events, live data,
    or broadly external facts, AND (b) `search_documents` and `search_conversation_history`
    have already been tried or are clearly irrelevant. Do NOT use this for questions
    that can be answered from the uploaded documents. Web search may be disabled by
    the user — calling it when disabled will return an error.

    Args:
        query: A concise, keyword-rich search query. Rephrase and retry with different
               keywords if initial results are off-topic or insufficient.
    """
    try:
        results = await asyncio.to_thread(search_tool.invoke, query)
        return str(results)
    except Exception as e:
        return f"Web search failed: {str(e)}"


@tool
async def ask_for_clarification(options: List[str]) -> str:
    """
    Present the user with 2–4 interpretations of their ambiguous question so they can
    select the one they intended.

    Only call this when the question is genuinely ambiguous AND making a reasonable
    assumption would risk answering the wrong question entirely. Do NOT use for minor
    phrasing uncertainty — make a safe assumption and proceed instead.

    Each option must be a complete, self-contained question representing a distinct
    interpretation of the user's intent, not just a short label or phrase.

    Args:
        options: A list of 2–4 complete questions, each representing a distinct and
                 plausible interpretation of what the user might have meant.
    """
    return json.dumps({"__clarification_options__": options})


@tool
async def list_uploaded_documents(config: RunnableConfig = None) -> str:
    """
    Return metadata for all PDF documents indexed in this thread:
    file name, file hash, and upload order (most recent first).

    Use when the user references "the document", "the PDF", "the first file",
    "the report", or any document by name or topic — to identify the correct
    file_hash before calling search_pdf_by_document. Do NOT call this on every
    request; only invoke it when you genuinely need to resolve a document reference.
    """
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id")
        if not thread_id:
            return "No thread context found."

        from database import get_thread_files
        files = await get_thread_files(thread_id)

        if not files:
            return "No documents are uploaded to this thread."

        doc_list = [
            {"index": i + 1, "file_name": f.file_name, "file_hash": f.file_hash}
            for i, f in enumerate(files)
        ]
        return json.dumps(doc_list, indent=2)
    except Exception as e:
        return f"Error listing documents: {e}"


@tool
async def search_pdf_by_document(
    query: str,
    file_hash: str,
    max_results: int = 8,
    config: RunnableConfig = None,
) -> str:
    """
    Semantic search scoped to a SINGLE document identified by file_hash.

    Use when the user explicitly refers to a specific document and you have resolved
    its file_hash via list_uploaded_documents (or from the document list in the
    pre-fetched context). Prefer this over search_documents when the target document
    is known — it avoids contaminating results with chunks from unrelated files.

    Args:
        query: Natural-language question to search for within the document.
        file_hash: The file_hash of the target document.
        max_results: Number of seed chunks before neighbor expansion.
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
        raw_chunks = await db.search_pdf_chunks(
            thread_id=thread_id,
            query_vector=query_vector,
            limit=max_results,
            file_hash=file_hash,
        )

        if not raw_chunks:
            return f"No relevant content found in document {file_hash}."

        expansion_radius = max(2, min(10, int(context_window / 8000) + 1))
        chunk_ids_to_fetch: set = set()
        for hit in raw_chunks:
            chunk_id = hit.get("chunk_id")
            if chunk_id is not None:
                for neighbor_id in range(chunk_id - expansion_radius, chunk_id + expansion_radius + 1):
                    if neighbor_id >= 0:
                        chunk_ids_to_fetch.add(neighbor_id)

        expanded_chunks = await db.get_chunks_by_ids(
            thread_id=thread_id,
            file_hash=file_hash,
            chunk_ids=list(chunk_ids_to_fetch),
        )
        expanded_chunks.sort(key=lambda x: x.get("chunk_id", 0))

        sources = []
        context_parts = []
        for chunk in expanded_chunks:
            text = chunk.get("text", "")
            context_parts.append(text)
            sources.append({
                "text": text[:200] + "..." if len(text) > 200 else text,
                "file_hash": file_hash,
                "score": chunk.get("score", 0.0),
            })

        return json.dumps({
            "content": "\n\n".join(context_parts),
            "__pdf_sources__": sources,
        })
    except Exception as e:
        logger.error(f"Error in search_pdf_by_document: {e}", exc_info=True)
        return f"Error searching document: {e}"


@tool
async def find_topic_anchor_in_history(
    topic: str,
    config: RunnableConfig = None,
) -> str:
    """
    Find the chronological FIRST occurrence of a topic in the conversation history.
    Returns the approximate turn number, message_id, and a short excerpt.

    Use for temporal references like "what you first said about X",
    "when we started discussing Y", or "your original answer about Z".
    This returns a precise chronological anchor so you can rewrite the query
    with temporal precision (e.g., "In the message at turn 3, what was stated about X?").

    Args:
        topic: The topic or question to locate in the conversation history.
    """
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id")
        embedding_model = conf.get("embedding_model")
        if not thread_id or not embedding_model:
            return "No thread context found."

        embed_model = get_embedding_model(embedding_model)
        query_vector = await invoke_with_retry(embed_model.aembed_query, topic)

        db = QdrantAdapter()
        recalled = await db.search_chat_memory(
            thread_id=thread_id,
            query_vector=query_vector,
            limit=5,
        )

        if not recalled:
            return "No relevant history found for this topic."

        from database import get_thread_messages
        all_messages = await get_thread_messages(thread_id, limit=2000)
        position_map = {msg.id: i + 1 for i, msg in enumerate(all_messages)}

        results = []
        for mem in recalled:
            msg_id = mem.get("message_id")
            turn_number = position_map.get(msg_id, "?")
            text = mem.get("text", "")
            excerpt = text[:300] + "..." if len(text) > 300 else text
            results.append({
                "message_id": msg_id,
                "turn_number": turn_number,
                "score": round(mem.get("score", 0.0), 3),
                "excerpt": excerpt,
            })

        # Sort ascending so the earliest occurrence is first
        results.sort(key=lambda x: x["turn_number"] if isinstance(x["turn_number"], int) else 9999)

        return json.dumps({"topic": topic, "anchors": results[:3]})
    except Exception as e:
        return f"Error finding topic anchor: {e}"


# ---------------------------------------------------------------------------
# Pre-fetch bundle formatter
# ---------------------------------------------------------------------------

def _format_prefetch_for_prompt(bundle: Optional[Dict[str, Any]]) -> str:
    """
    Format a pre-fetched context bundle as a labelled block for injection into
    LLM system prompts.  Returns an empty string when the bundle is None/empty.
    """
    if not bundle:
        return ""

    parts: List[str] = []

    stats = bundle.get("stats", {})
    if stats and stats.get("total_messages", 0) > 0:
        parts.append(
            f"[CONVERSATION STATS]  Total turns: {stats.get('total_messages', 0)} | "
            f"Estimated tokens in history: {int(stats.get('estimated_history_tokens', 0))}"
        )

    docs = bundle.get("documents", [])
    if docs:
        doc_lines = "\n".join(
            [f"  {d['index']}. {d['file_name']}  (file_hash: {d['file_hash']})" for d in docs]
        )
        parts.append(f"[UPLOADED DOCUMENTS — {len(docs)} file(s)]\n{doc_lines}")

    recent = bundle.get("recent_history_text", "")
    if recent:
        parts.append(f"[RECENT CONVERSATION TURNS]\n{recent}")

    semantic = bundle.get("semantic_history_text", "")
    if semantic:
        parts.append(f"[SEMANTICALLY RELEVANT PAST QA PAIRS]\n{semantic}")

    pdf = bundle.get("pdf_evidence_text", "")
    if pdf:
        parts.append(f"[PDF DOCUMENT EVIDENCE  (queried with raw/un-rewritten question)]\n{pdf}")

    if not parts:
        return ""

    sep = "═" * 64
    return (
        f"\n\n{sep}\n"
        "PRE-FETCHED CONTEXT  (assembled before this call — no tool calls needed for this data):\n"
        f"{sep}\n"
        + "\n\n".join(parts)
        + f"\n{sep}\n"
        "NOTE: PDF Evidence and Semantic History were retrieved with the raw question.\n"
        "A better-rewritten query will improve precision — call tools ONLY when this\n"
        "pre-fetched context is genuinely insufficient to answer the user's request.\n"
        f"{sep}"
    )


# --- Intent Agent ---
class IntentAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str
    llm_model: str
    context_window: int
    iteration_count: int
    max_iterations: int
    intent_result: Optional[Dict[str, Any]]
    pre_fetch_bundle: Optional[Dict[str, Any]]

# Intent Agent tools:
# - Removed: get_conversation_stats, page_conversation_history, search_documents,
#   get_recent_context  (all covered by the pre-fetched bundle injected in the prompt)
# - Added:   list_uploaded_documents (resolve doc references),
#            find_topic_anchor_in_history (temporal anchoring)
# - Kept:    search_conversation_history (deep semantic recall beyond bundle),
#            ask_for_clarification (ambiguity resolution)
intent_tools_list = [
    search_conversation_history,
    list_uploaded_documents,
    find_topic_anchor_in_history,
    ask_for_clarification,
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
        f"""You are an expert query analyst. Your sole task is to classify the user's message and rewrite it into a self-contained, retrieval-optimized question.

        CONTEXT: Conversation history appears in the messages below AND as pre-fetched data at the end of this
        system prompt (stats, recent turns, semantic history, PDF evidence, document list). Use all of it
        directly — you do NOT need tools to retrieve recent history, stats, or PDF evidence that is already there.

        RUNTIME CONSTRAINTS:
        Your maximum context window is {context_window} tokens. Stay within this budget.

        YOUR OUTPUT — respond ONLY with this JSON object, no preamble or explanation:
        {{
          "status": "CLEAR_STANDALONE" | "CLEAR_FOLLOWUP" | "AMBIGUOUS",
          "rewritten_query": "A single, complete, natural-language question",
          "reference_type": "NONE" | "SEMANTIC" | "TEMPORAL" | "ENTITY",
          "context_coverage": "SUFFICIENT" | "PROBABLY_SUFFICIENT" | "INSUFFICIENT",
          "clarification_options": ["Full question for interpretation A", "Full question for interpretation B"] | null
        }}

        reference_type meanings:
          NONE     — standalone question, no prior-context reference
          SEMANTIC — user refers to a topic/concept discussed before ("what did we say about X")
          TEMPORAL — user refers to a chronological position ("your first answer", "at the start")
          ENTITY   — user refers to a specific named thing described earlier ("that figure", "the method")

        context_coverage meanings:
          SUFFICIENT          — pre-fetched context is almost certainly enough to answer; Orchestrator may skip tool calls
          PROBABLY_SUFFICIENT — partial match; Orchestrator should verify with one targeted tool call
          INSUFFICIENT        — clear gap; Orchestrator must retrieve more data

        STEP 1 — REWRITE THE QUERY (mandatory for all statuses):
        Expand the user's message into a fully self-contained question that performs well as a semantic search query.
        Even clear questions benefit from expansion — add specificity, resolve pronouns, and include relevant domain context.
        Write a single, natural question. Never prefix with "Q:" or use a "Q: ... A: ..." format.

        Good rewrites:
        - "What is RAG?" → "What is Retrieval-Augmented Generation (RAG) and how does it work?"
        - "Explain transformers" → "Explain the transformer architecture in machine learning, including self-attention mechanisms"
        - "What are the main findings?" → "What are the main findings or conclusions presented in the uploaded document?"
        - "How does it work?" (after discussing a document's method) → "How does the methodology described in the document work, step by step?"
        - "Tell me more" (after discussing a specific section) → "Provide a more detailed explanation of [the specific topic just discussed]"
        - "Summarize it" (after user uploads a PDF) → "Provide a comprehensive summary of the uploaded document"

        STEP 2 — CLASSIFY:
        - CLEAR_STANDALONE: The question is self-contained and its intent is unambiguous
        - CLEAR_FOLLOWUP: The question references earlier context but its intent is clear from the inline history
        - AMBIGUOUS: The question has multiple plausible interpretations that the inline history cannot resolve

        STEP 3 — TOOL USE POLICY:
        Pre-fetched context (stats, recent turns, semantic history, PDF evidence, document list) is already in
        the PRE-FETCHED CONTEXT block at the end of this prompt. Use it directly — no tool needed for that data.
        Only call tools for information that is NOT present or clearly insufficient in the pre-fetched block.

        AVAILABLE TOOLS AND WHEN TO USE:
        - `search_conversation_history` — ONLY if the pre-fetched semantic history block is absent or misses
          critical context required to rewrite an ambiguous follow-up.
        - `list_uploaded_documents` — ONLY when the user references a document by name/topic AND the document
          list in the pre-fetched block does not identify it.
        - `find_topic_anchor_in_history` — ONLY for TEMPORAL references ("your first answer about X", "when we
          started discussing Y") that cannot be resolved from the pre-fetched content.
        - `ask_for_clarification` — ONLY for genuinely ambiguous questions where a wrong assumption would answer
          an entirely different question. Each option must be a distinct, complete, self-contained question.
        - Do NOT call any tool on a clear standalone first message — output the JSON immediately."""
    )

    # Inject pre-fetched context bundle so the LLM has everything in its system prompt
    bundle = state.get("pre_fetch_bundle")
    if bundle:
        system_prompt += _format_prefetch_for_prompt(bundle)

    # For first messages, reinforce no-tool usage
    if is_first_message:
        system_prompt += (
            "\n\nFIRST MESSAGE: This is the opening message — there is no prior conversation history. "
            "Do NOT call any tools. Classify as CLEAR_STANDALONE, rewrite the query for retrieval clarity, "
            "and output the JSON immediately."
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
            f"Tool iteration budget exhausted (context window: {context_window} tokens). Do NOT call any tools.\n"
            "Using only the conversation history already in your context, rewrite the user's question "
            "into a standalone, retrieval-optimized form.\n"
            "Respond ONLY with valid JSON:\n"
            '{{"status": "CLEAR_STANDALONE" | "CLEAR_FOLLOWUP", '
            '"rewritten_query": "<rewritten question>", '
            '"reference_type": "NONE" | "SEMANTIC" | "TEMPORAL" | "ENTITY", '
            '"context_coverage": "PROBABLY_SUFFICIENT", '
            '"clarification_options": null}}'
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
                "reference_type": "NONE",
                "context_coverage": "PROBABLY_SUFFICIENT",
                "clarification_options": None
            }
        return {"messages": [response], "iteration_count": iteration, "intent_result": intent_result}
    except Exception as e:
        logger.error(f"Force intent answer LLM call failed: {e}")
        intent_result = {
            "status": "CLEAR_STANDALONE",
            "rewritten_query": original_question,
            "reference_type": "NONE",
            "context_coverage": "PROBABLY_SUFFICIENT",
            "clarification_options": None
        }
        fallback_msg = AIMessage(content=json.dumps(intent_result))
        return {"messages": [fallback_msg], "iteration_count": iteration, "intent_result": intent_result}

class IntentToolNode(ToolNode):
    async def ainvoke(self, input: dict, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        res = await super().ainvoke(input, config, **kwargs)
        messages = res.get("messages", [])
        
        # Check if ask_for_clarification was called
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

# Orchestrator tools list:
# - Removed: get_conversation_stats, page_conversation_history, get_recent_context
#   (all covered by the pre-fetched bundle injected in the system prompt)
# - Added:   search_pdf_by_document (scoped single-document search)
# - Kept:    search_documents (re-run with rewritten query for better precision),
#            search_conversation_history, search_web, ask_for_clarification
tools_list = [
    search_documents,
    search_pdf_by_document,
    search_conversation_history,
    search_web,
    ask_for_clarification,
]


TOOL_FRIENDLY_CONFIG = {
    "search_documents": {
        "id": "document_evidence",
        "display_name": "Document Evidence",
        "description": "Semantic vector search across ALL uploaded PDF documents — returns matching chunks with surrounding context.",
        "default_prompt": "Use when the question spans multiple documents or the target document is unknown. If results are weak, retry with a rephrased or more specific query. When the document is known, prefer search_pdf_by_document instead.",
    },
    "search_pdf_by_document": {
        "id": "focused_document_evidence",
        "display_name": "Focused Document Evidence",
        "description": "Semantic search scoped to a SINGLE document by file_hash — avoids contamination from unrelated files.",
        "default_prompt": "Prefer this over Document Evidence when the user explicitly refers to a specific document. Resolve the file_hash from the document list in the pre-fetched context or via list_uploaded_documents.",
    },
    "search_conversation_history": {
        "id": "deep_memory",
        "display_name": "Deep Memory",
        "description": "Semantic vector search across all past conversation QA pairs in this thread.",
        "default_prompt": "Use for thematic or semantic recall (e.g., 'what did we discuss about X?') across the full thread history. Retry with a rephrased query if initial results are irrelevant.",
    },
    "search_web": {
        "id": "live_web_recon",
        "display_name": "Internet Search",
        "description": "Search the web for external, real-time, or post-training knowledge.",
        "default_prompt": "Last resort for external or live information. Exhaust Document Evidence and Deep Memory first. Rephrase and retry with different keywords if initial results are off-topic.",
    },
    "ask_for_clarification": {
        "id": "clarify_intent",
        "display_name": "Clarify Intent",
        "description": "Present the user with distinct interpretations of an ambiguous question.",
        "default_prompt": "Use only when the question has multiple plausible interpretations and making an assumption risks answering the wrong question entirely. Each option must be a complete, self-contained question.",
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
    valid_tools = tools_list if state.get("use_web_search", False) else [t for t in tools_list if t.name != "search_web"]
    llm_with_tools = llm.bind_tools(valid_tools)
    
    context_window = state.get('context_window', DEFAULT_TOKEN_BUDGET)
    system_role = sanitize_system_role(state.get("system_role", ""))
    tool_instructions = normalize_tool_instructions(state.get("tool_instructions", {}))
    custom_instructions = sanitize_custom_instructions(state.get("custom_instructions", ""))

    prompt_content = build_system_prompt(
        context_window=context_window,
        system_role=system_role,
        tool_instructions=tool_instructions,
        custom_instructions=custom_instructions
    )

    # Inject pre-fetched context bundle + pre-fetch-first retrieval policy
    bundle = state.get("pre_fetch_bundle")
    if bundle:
        bundle_text = _format_prefetch_for_prompt(bundle)
        if bundle_text:
            prompt_content += (
                "\n\nPRE-FETCH RETRIEVAL POLICY (LOCKED):\n"
                "Pre-fetched context (recent turns, semantic history, PDF evidence, document list) is\n"
                "already present in the PRE-FETCHED CONTEXT block below. Before calling any tool:\n"
                "1. Assess whether the pre-fetched content answers the rewritten query with confidence.\n"
                "   If YES — answer directly; call no tool.\n"
                "2. If PDF evidence is present but the rewritten query is more specific than the raw question:\n"
                "   call search_documents or search_pdf_by_document ONCE with the rewritten query.\n"
                "3. If the question targets a specific document and its file_hash is in the document list:\n"
                "   prefer search_pdf_by_document (scoped) over search_documents (all documents).\n"
                "4. Do NOT call search_conversation_history just to re-read recent turns — the recent\n"
                "   conversation and semantic history are already in the pre-fetched block.\n"
            ) + bundle_text

    sys_prompt = SystemMessage(content=prompt_content)

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
                "1. For questions requiring specific facts, document content, or prior conversation context, use tools rather than relying on internal knowledge alone.",
                "2. Check the PRE-FETCHED CONTEXT block in this prompt before calling any tool — it already contains recent history, semantic memory, PDF evidence, and the document list.",
                "3. Tool priority when pre-fetched context is insufficient: Focused Document Evidence (known doc) → Document Evidence (all docs) → Deep Memory → Internet Search.",
                "4. If a tool returns weak or empty results, retry with a rephrased or more specific query before concluding the information is absent.",
                "5. You may make multiple tool calls across iterations; avoid redundant calls with nearly identical queries.",
                "6. Internet Search is only available when explicitly enabled by the user — do not call it otherwise.",
                "7. Clarify Intent is only for genuinely ambiguous questions where a wrong assumption would lead to an entirely wrong answer.",
                "8. If custom user instructions conflict with this locked contract, follow this locked contract."
            ])
        ),
        (
            "TOOL PLAYBOOK (USER-CONFIGURABLE)",
            "\n".join([f"- {item['tool_name']}: {playbook.get(item['id'], item['default_prompt'])}" for item in catalog]),
        ),
        (
            "ANSWER POLICY (LOCKED)",
            "\n".join([
                "1. Output a final answer only after retrieving sufficient context through tool calls.",
                "2. Synthesize retrieved content into a coherent, well-structured answer — do not paste raw chunks verbatim.",
                "3. If retrieved information is partial or uncertain, explicitly state that limitation rather than fabricating details.",
                "4. Do NOT prefix your response with 'A:' or mimic the 'Q: ... A: ...' pattern. Answer directly.",
            ])
        ),
    ]
    if custom_instructions:
        sections.append(("USER CUSTOM INSTRUCTIONS (EDITABLE)", custom_instructions))

    return "\n\n".join([f"{title}:\n{body}" for title, body in sections])
