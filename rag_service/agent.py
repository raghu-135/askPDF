import asyncio
import logging
import json
from typing import TypedDict, List, Annotated, Dict, Any, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from models import get_llm, get_embedding_model, DEFAULT_TOKEN_BUDGET, DEFAULT_MAX_ITERATIONS
from prompt_loaders import (
    get_orchestrator_prompt,
    get_orchestrator_prompt_compact,
    get_orchestrator_phase0_prompt,
    get_orchestrator_phase0_prompt_compact,
    get_intent_agent_prompt,
    get_intent_agent_prompt_compact,
    get_web_search_mandate,
)
from agent_helpers import (
    build_chat_prompt,
    parse_intent_response,
    looks_like_followup,
    evidence_insufficient,
    collect_tool_sources,
)
from prompt_defaults import DEFAULT_SYSTEM_ROLE
from intent_fallback import heuristic_rewrite_query
from vectordb.qdrant import get_qdrant
from retrieval import fetch_semantic_history, get_document_name_lookup, group_document_chunks

logger = logging.getLogger(__name__)

search_tool = DuckDuckGoSearchResults(output_format="list", num_results=6)

async def invoke_with_retry(func, *args, **kwargs):
    max_retries = 10
    base_delay = 2
    for i in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            # 503 – model still loading / service unavailable
            is_503_loading = "503" in err_str and ("loading" in err_str or "unavailable" in err_str)
            # 400 – LM Studio unloaded the model between requests
            is_model_unloaded = "400" in err_str and "model unloaded" in err_str
            # 404 - Model not found / not currently loaded in memory
            is_model_not_found = "404" in err_str and ("not found" in err_str or "model" in err_str)
            # 500 - Generic load errors or out of memory briefly on LLM server
            is_500_loading = "500" in err_str and ("load" in err_str or "memory" in err_str)
            # 429 – rate limit / too many requests
            is_rate_limit = "429" in err_str or "rate limit" in err_str or "too many requests" in err_str

            if is_503_loading or is_model_unloaded or is_rate_limit or is_model_not_found or is_500_loading:
                reason = (
                    "Model is loading (503)" if is_503_loading
                    else "Model was unloaded (400)" if is_model_unloaded
                    else "Model not loaded/found (404)" if is_model_not_found
                    else "Temporary model failure/OOM (500)" if is_500_loading
                    else "Rate limited (429)"
                )
                
                delay = base_delay * (2 ** min(i, 4)) # Exponential backoff up to 32s max delay
                logger.warning(f"{reason}. Retrying in {delay}s... (Attempt {i+1}/{max_retries})")
                await asyncio.sleep(delay)
                continue
            raise
    raise Exception("Max retries reached while waiting for model to become available.")

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str
    llm_model: str
    embedding_model: str
    context_window: int
    use_web_search: bool
    document_sources: List[Dict[str, Any]]
    web_sources: List[Dict[str, Any]]
    used_chat_ids: List[str]
    clarification_options: Optional[List[str]]
    iteration_count: int
    max_iterations: int
    system_role: str
    tool_instructions: Dict[str, str]
    custom_instructions: str
    pre_fetch_bundle: Optional[Dict[str, Any]]
    intent_agent_ran: bool  # True when Intent Agent preprocessed the query; False = Orchestrator self-preprocesses
    reasoning_mode: bool
    working_query: str
    intent_reference_type: str




@tool
async def search_documents(query: str, max_results: int = 10, config: RunnableConfig = None) -> str:
    """
    Perform a semantic vector search over all uploaded documents (PDFs + webpages) AND previously cached
    internet search results for this thread.
    Returns the most relevant chunks along with neighboring context passages for continuity.

    This uses embedding-based similarity — phrase queries as natural questions rather than
    keyword strings for best results. If the first call returns weak or irrelevant evidence,
    retry with a rephrased or more specific query before concluding the information is absent.

    Each returned passage is prefixed with its source so you can cite it accurately:
      - Document passages: [Source: PDF: <filename>] or [Source: Webpage: <title> | <url>]
      - Cached web results: [Source: Internet Search — "<title>" | <url>]

    Args:
        query: A natural-language question or description of the fact to locate.
        max_results: Number of seed chunks to retrieve before context expansion.
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

        db = get_qdrant()

        # ── Build a file_hash → file_name lookup from thread_stats (no DB join) ──
        hash_to_name = await get_document_name_lookup(thread_id)

        # ── Document chunk search with neighbor expansion ──
        raw_doc_chunks = await db.search_knowledge_sources(
            thread_id=thread_id,
            query_vector=query_vector,
            limit=max_results
        )

        expansion_radius = max(2, min(10, int(context_window / 8000) + 1))
        file_chunk_map = {}
        for hit in raw_doc_chunks:
            file_hash = hit.get("file_hash")
            chunk_id = hit.get("chunk_id")
            if file_hash is not None and chunk_id is not None:
                if file_hash not in file_chunk_map:
                    file_chunk_map[file_hash] = set()
                for neighbor_id in range(chunk_id - expansion_radius, chunk_id + expansion_radius + 1):
                    if neighbor_id >= 0:
                        file_chunk_map[file_hash].add(neighbor_id)

        expanded_doc_chunks = []
        for file_hash, id_set in file_chunk_map.items():
            expanded_batch = await db.get_knowledge_source_chunks_by_ids(
                thread_id=thread_id,
                file_hash=file_hash,
                chunk_ids=list(id_set)
            )
            expanded_doc_chunks.extend(expanded_batch)

        expanded_doc_chunks.sort(key=lambda x: (x.get("file_hash", ""), x.get("chunk_id", 0)))

        # ── Cached web search results ──
        web_chunks = await db.search_web_chunks(
            thread_id=thread_id,
            query_vector=query_vector,
            limit=max(3, max_results // 3),
        )

        if not expanded_doc_chunks and not web_chunks:
            return "No relevant content found in documents or cached web results."

        document_sources = []
        web_sources = []
        context_parts = []

        # Group document chunks by file to reduce context window bloat
        document_context, document_sources = group_document_chunks(expanded_doc_chunks, hash_to_name)
        if document_context:
            context_parts.append(document_context)

        # Group cached web chunks by URL
        web_groups = {}
        for wchunk in web_chunks:
            url = wchunk.get("url", "")
            if url not in web_groups:
                web_groups[url] = {"title": wchunk.get("title", url), "texts": []}
            web_groups[url]["texts"].append(wchunk.get("text", ""))

            web_sources.append({
                "text": wchunk.get("text", "")[:200] + "...",
                "url": url,
                "title": wchunk.get("title", url),
                "score": wchunk.get("score", 0.0),
            })

        for url, group in web_groups.items():
            combined_text = "\n".join(group["texts"])
            context_parts.append(f'[Source: Internet Search — "{group["title"]}" | {url}]\n{combined_text}')

        result: Dict[str, Any] = {"content": "\n\n".join(context_parts)}
        if document_sources:
            result["__document_sources__"] = document_sources
        if web_sources:
            result["__web_sources__"] = web_sources
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error in search_documents: {e}", exc_info=True)
        return f"Error retrieving knowledge: {e}"


@tool
async def search_conversation_history(query: str, max_results: int = 10, config: RunnableConfig = None) -> str:
    """
    Perform a semantic vector search across all past conversation QA pairs in this thread.
    Returns the most relevant past exchanges ranked by embedding similarity, regardless
    of when they occurred in the conversation.

    Use this for thematic or conceptual recall (e.g., "what did we previously discuss
    about X?"). Searches by embedding similarity regardless of when in the conversation
    the content appeared. Retry with a rephrased query if initial results are off-topic.

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
        
        db = get_qdrant()
        history, used_ids = await fetch_semantic_history(
            thread_id=thread_id,
            query_vector=query_vector,
            limit=max_results,
        )

        if not history:
            return "No relevant past conversations found."

        return json.dumps({
            "content": history,
            "__used_chat_ids__": used_ids
        })
    except Exception as e:
        return f"Error retrieving chat memory: {e}"


def _normalize_web_results(raw: Any, query: str) -> List[Dict[str, str]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return [{"snippet": raw, "title": query, "link": ""}]
    return []


async def _run_web_search(query: str, max_results: Optional[int]) -> Optional[Dict[str, List[str]]]:
    raw = await asyncio.to_thread(search_tool.invoke, query)
    logger.info(f"Raw search results for '{query}': {raw}")
    results_list = _normalize_web_results(raw, query)
    if not results_list:
        return None

    texts = [r.get("snippet", r.get("body", "")) for r in results_list]
    urls = [r.get("link", r.get("href", "")) for r in results_list]
    titles = [r.get("title", "") for r in results_list]

    valid = [(t, u, ti) for t, u, ti in zip(texts, urls, titles) if t.strip()]
    if not valid:
        return None

    texts, urls, titles = zip(*valid)
    texts = list(texts)
    urls = list(urls)
    titles = list(titles)
    if max_results is not None:
        texts = texts[:max_results]
        urls = urls[:max_results]
        titles = titles[:max_results]
    return {"texts": texts, "urls": urls, "titles": titles}


def _format_web_context(texts: List[str], urls: List[str], titles: List[str]) -> Dict[str, Any]:
    web_groups: Dict[str, Dict[str, Any]] = {}
    web_sources: List[Dict[str, str]] = []
    for text, url, title in zip(texts, urls, titles):
        if url not in web_groups:
            web_groups[url] = {"title": title or url or "Internet Search", "texts": []}
        web_groups[url]["texts"].append(text)
        web_sources.append({
            "text": text[:200] + "...",
            "url": url,
            "title": title or "Internet Search",
        })

    context_parts = []
    for url, group in web_groups.items():
        combined_text = "\n".join(group["texts"])
        context_parts.append(f'[Source: Internet Search — "{group["title"]}" | {url}]\n{combined_text}')

    return {
        "content": "\n\n".join(context_parts),
        "__web_sources__": web_sources,
    }


@tool
async def search_web(query: str, config: RunnableConfig = None) -> str:
    """
    Search the web for external, real-time, or post-training knowledge.

    Results are automatically stored in the thread's knowledge base so that future
    questions on the same topic can be answered without a new web request.

    Use this along with search_documents when you need to augment the knowledge from
    uploaded documents with the latest information from the internet. This helps
    provide a more comprehensive answer by checking both internal documents and external
    web resources in parallel.

    If internet search is not enabled for this session, this tool will return a
    message indicating that and no search will be performed.

    Each returned passage is prefixed with its source URL so you can cite it accurately:
      [Source: Internet Search — "<title>" | <url>]

    Args:
        query: A concise, keyword-rich search query. Rephrase and retry with different
               keywords if initial results are off-topic or insufficient.
    """
    try:
        conf = config.get("configurable", {}) if config else {}
        if not conf.get("use_web_search", False):
            return "Internet search is not enabled for this session. The user has not turned on web search, so no internet results are available. Answer using only the uploaded documents and conversation history."

        logger.info(f"--- WEB SEARCH INITIATED --- Query: '{query}'")
        thread_id = conf.get("thread_id")
        embedding_model = conf.get("embedding_model")

        result = await _run_web_search(query, max_results=6)
        if not result:
            return "Web search returned no usable text."

        texts = result["texts"]
        urls = result["urls"]
        titles = result["titles"]

        # ── Persist results in Qdrant for future retrieval ──
        if thread_id and embedding_model and conf.get("web_search_index", True):
            try:
                from rag import index_web_search_for_thread
                asyncio.create_task(
                    index_web_search_for_thread(
                        thread_id=thread_id,
                        query=query,
                        texts=texts,
                        urls=urls,
                        titles=titles,
                        embedding_model_name=embedding_model,
                    )
                )
            except Exception as idx_err:
                logger.warning(f"Web search indexing skipped: {idx_err}")

        return json.dumps(_format_web_context(texts, urls, titles))
    except Exception as e:
        logger.error(f"Web search failed: {e}", exc_info=True)
        return f"Web search failed: {str(e)}"


@tool
async def search_web_intent(query: str, config: RunnableConfig = None) -> str:
    """
    Lightweight web lookup for query rewriting, intent disambiguation, and
    time-sensitivity detection.

    Use ONLY to identify unknown terms/entities or determine if the question is about
    current events, prices, or other time-sensitive facts. Use results only to
    clarify user intent and improve the rewritten query. Do NOT use this as evidence
    in the final answer.

    If internet search is not enabled for this session, this tool will return a
    message indicating that and no search will be performed.

    Args:
        query: A concise query aimed at identifying the term or entity in question.
    """
    try:
        conf = config.get("configurable", {}) if config else {}
        if not conf.get("use_web_search", False):
            return "Internet search is not enabled for this session. The user has not turned on web search, so no internet results are available."

        logger.info(f"--- INTENT WEB SEARCH INITIATED --- Query: '{query}'")
        result = await _run_web_search(query, max_results=None)
        if not result:
            return "Web search returned no usable text."

        return json.dumps(_format_web_context(result["texts"], result["urls"], result["titles"]))
    except Exception as e:
        logger.error(f"Intent web search failed: {e}", exc_info=True)
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
    Return metadata for all documents indexed in this thread:
    file name, file hash, and upload order (most recent first).

    Use when the user references "the document", "the PDF", "the webpage", "the first file",
    "the report", or any document by name or topic — to identify the correct
    file_hash before calling search_document_by_id. Do NOT call this on every
    request; only invoke it when you genuinely need to resolve a document reference.
    """
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id")
        if not thread_id:
            return "No thread context found."

        from database import get_thread_shape as _get_shape
        shape = await _get_shape(thread_id)
        docs = shape["documents"]

        if not docs:
            return "No documents are uploaded to this thread."

        doc_list = [
            {
                "index": i + 1,
                "file_name": meta["file_name"],
                "file_hash": fh,
                "source_type": meta.get("source_type", "pdf"),
                "chunks": meta.get("chunk_count", 0),
                "status": meta.get("indexing_status", "unknown"),
            }
            for i, (fh, meta) in enumerate(docs.items())
        ]
        return json.dumps(doc_list, indent=2)
    except Exception as e:
        return f"Error listing documents: {e}"


@tool
async def search_document_by_id(
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

        db = get_qdrant()
        raw_chunks = await db.search_knowledge_sources(
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

        expanded_chunks = await db.get_knowledge_source_chunks_by_ids(
            thread_id=thread_id,
            file_hash=file_hash,
            chunk_ids=list(chunk_ids_to_fetch),
        )
        expanded_chunks.sort(key=lambda x: x.get("chunk_id", 0))

        # Resolve file name for source attribution from thread_stats (no DB join)
        hash_to_name = await get_document_name_lookup(thread_id)
        fname = hash_to_name.get(file_hash, file_hash)

        document_context, sources = group_document_chunks(expanded_chunks, {file_hash: fname})
        if not document_context:
            document_context = ""

        return json.dumps({
            "content": document_context,
            "__document_sources__": sources,
        })
    except Exception as e:
        logger.error(f"Error in search_document_by_id: {e}", exc_info=True)
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

        db = get_qdrant()
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

    documents = bundle.get("document_evidence_text", "")
    if documents:
        parts.append(f"[DOCUMENT EVIDENCE (PDF + WEBPAGE)  (queried with raw/un-rewritten question)]\n{documents}")

    web = bundle.get("web_evidence_text", "")
    if web:
        parts.append(f"[WEB SEARCH EVIDENCE]\n{web}")

    if not parts:
        return ""

    sep = "=" * 64
    return (
        f"\n\n{sep}\n"
        "PRE-FETCHED CONTEXT  (assembled before this call — no tool calls needed for this data):\n"
        f"{sep}\n"
        + "\n\n".join(parts)
        + f"\n{sep}\n"
        "NOTE: Document Evidence and Semantic History were retrieved with the raw question.\n"
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
    reasoning_mode: bool
    intent_tools_used: bool
    use_web_search: bool



async def call_intent_model(state: IntentAgentState, config: RunnableConfig):
    """
    Single-pass Intent Agent: classifies and rewrites the user's query for the Orchestrator.
    No tools, no retries — one LLM call, one JSON output.
    """
    messages = state["messages"]
    llm = get_llm(state["llm_model"], temperature=0.0)
    allow_intent_web_search = bool(state.get("use_web_search"))
    
    tools_to_bind = intent_tools.copy() if allow_intent_web_search else []
    
    if tools_to_bind:
        llm_with_tools = llm.bind_tools(tools_to_bind)
    else:
        llm_with_tools = llm
    iteration = state.get("iteration_count", 0) + 1
    context_window = state.get("context_window", DEFAULT_TOKEN_BUDGET)
    reasoning_mode = state.get("reasoning_mode", True)

    # Load and format the Intent Agent prompt
    bundle = state.get("pre_fetch_bundle")
    prefetch_text = _format_prefetch_for_prompt(bundle) if bundle else ""

    base_prompt = get_intent_agent_prompt() if reasoning_mode else get_intent_agent_prompt_compact()
    system_prompt = base_prompt.replace("{PREFETCH_CONTEXT}", prefetch_text)
    if not allow_intent_web_search:
        system_prompt += "\n\nWeb search is disabled for this session. Do NOT call any web search tools."
    prompt_template = build_chat_prompt()
    input_messages = prompt_template.format_messages(
        system_prompt=system_prompt,
        messages=messages if not (messages and isinstance(messages[0], SystemMessage)) else messages[1:],
    )

    # Log complete prompt for Intent Agent in OpenAI-like format
    logger.info(f"--- INTENT AGENT PROMPT BEGIN [thread_id: {state.get('thread_id')}] ---")
    payload = []
    for msg in input_messages:
        role = "system" if isinstance(msg, SystemMessage) else "user" if isinstance(msg, HumanMessage) else "assistant"
        entry = {"role": role, "content": msg.content}
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            entry["tool_calls"] = msg.tool_calls
        if isinstance(msg, ToolMessage):
            entry["role"] = "tool"
            entry["tool_call_id"] = msg.tool_call_id
        payload.append(entry)
    logger.info(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info(f"--- INTENT AGENT PROMPT END ---")

    # Single direct call — no tools, minimal retries
    try:
        response = await invoke_with_retry(llm_with_tools.ainvoke, input_messages)
    except Exception as e:
        logger.error(f"Intent Agent LLM call failed: {e}")
        original_question = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        intent_result = {
            "route": "ANSWER",
            "rewritten_query": original_question,
            "reference_type": "NONE",
            "context_coverage": "PARTIAL",
            "clarification_options": None,
        }
        return {"messages": [AIMessage(content=json.dumps(intent_result))], "iteration_count": iteration, "intent_result": intent_result}

    if getattr(response, "tool_calls", None):
        # If it reaches here, it's calling search_web_intent
        return {
            "messages": [response],
            "iteration_count": iteration,
            "intent_result": None,
        }

    intent_result = parse_intent_response(response.content, logger=logger)
    if intent_result is None:
        logger.warning("Intent XML invalid; retrying once with strict XML instruction.")
        retry_msg = HumanMessage(
            content="Your previous output was invalid or missing required XML tags. Output the required XML tags: <route>, <rewritten_query>, <reference_type>, <context_coverage>."
        )
        retry_messages = input_messages + [retry_msg]
        try:
            retry_response = await invoke_with_retry(llm_with_tools.ainvoke, retry_messages)
            intent_result = parse_intent_response(retry_response.content, logger=logger)
            response = retry_response
        except Exception as e:
            logger.error(f"Intent Agent retry failed: {e}")

    if intent_result is None:
        original_question = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        fallback_query = heuristic_rewrite_query(original_question, state.get("pre_fetch_bundle"))
        intent_result = {
            "route": "ANSWER",
            "rewritten_query": fallback_query,
            "reference_type": "NONE",
            "context_coverage": "INSUFFICIENT",
            "clarification_options": None,
        }
        logger.warning("Intent XML invalid after retry; using heuristic fallback.")

    return {
        "messages": [response],
        "iteration_count": iteration,
        "intent_result": intent_result,
    }




class IntentToolNode(ToolNode):
    async def ainvoke(self, input: dict, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        res = await super().ainvoke(input, config, **kwargs)
        return {
            "messages": res.get("messages", []),
            "intent_tools_used": True,
        }


def intent_should_continue(state: IntentAgentState):
    messages = state["messages"]
    last_message = messages[-1]
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 1)
    tools_used = state.get("intent_tools_used", False)

    if getattr(last_message, "tool_calls", None):
        if tools_used and iteration_count >= max_iterations:
            return END
        return "tools"
    return END


intent_tools = [
    search_web_intent,
]

intent_workflow = StateGraph(IntentAgentState)
intent_workflow.add_node("agent", call_intent_model)
intent_workflow.add_node("tools", IntentToolNode(intent_tools))
intent_workflow.add_edge(START, "agent")
intent_workflow.add_conditional_edges(
    "agent",
    intent_should_continue,
    {"tools": "tools", END: END},
)
intent_workflow.add_edge("tools", "agent")

intent_app = intent_workflow.compile()
# --- End Intent Agent ---

@tool
async def get_thread_shape(config: RunnableConfig = None) -> str:
    """
    Return a compact snapshot of this thread's content inventory: number of
    uploaded documents/websites with their chunk counts and indexing status,
    plus QA history volume (total pairs and average size).

    Use this to calibrate your retrieval strategy BEFORE making tool calls:
    - Documents with large chunk_count → deep retrieval likely needed
    - Many QA pairs with high avg_qa_chars → rich semantic memory available
    - Documents with status 'pending' or 'failed' → indexing incomplete, warn user

    This reads from a pre-maintained stats table — it is very fast and does NOT
    perform any vector search or scan the messages table.
    """
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id")
        if not thread_id:
            return "No thread context found."

        from database import get_thread_shape as _get_shape
        shape = await _get_shape(thread_id)

        qa_pairs = shape["total_qa_pairs"]
        avg_qa = shape["avg_qa_chars"]
        total_qa = shape["total_qa_chars"]
        docs = shape["documents"]

        lines = ["[THREAD SHAPE]"]
        lines.append(
            f"QA History  : {qa_pairs} pair(s) | {avg_qa:,.0f} avg chars/pair | {total_qa:,} total chars"
        )
        if docs:
            lines.append(f"Documents   : {len(docs)} source(s)")
            for i, (fh, meta) in enumerate(docs.items(), start=1):
                status = meta.get("indexing_status", "unknown")
                chunks = meta.get("chunk_count", 0)
                chars = meta.get("total_chars", 0)
                name = meta.get("file_name", fh)
                stype = meta.get("source_type", "pdf")
                tag = f"[{stype}]" if stype != "pdf" else ""
                lines.append(
                    f"  {i}. {name} {tag}  →  {chunks} chunks | {chars:,} chars | {status}"
                )
        else:
            lines.append("Documents   : none uploaded yet")

        return "\n".join(lines)
    except Exception as e:
        return f"Error reading thread shape: {e}"


tools_list = [
    get_thread_shape,
    search_documents,
    search_document_by_id,
    search_conversation_history,
    find_topic_anchor_in_history,
    search_web,
    ask_for_clarification,
]


TOOL_FRIENDLY_CONFIG = {
    "search_documents": {
        "id": "document_evidence",
        "display_name": "Document Evidence",
        "description": "Semantic vector search across ALL uploaded documents (PDFs + webpages) — returns matching chunks with surrounding context.",
        "default_prompt": "Use when the question spans multiple documents or the target document is unknown. If results are weak, retry with a rephrased or more specific query. When the document is known, prefer search_document_by_id instead.",
    },
    "search_document_by_id": {
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
    "find_topic_anchor_in_history": {
        "id": "temporal_anchor",
        "display_name": "Temporal Anchor",
        "description": "Locates the chronological FIRST occurrence of a topic in the conversation history, returning turn number, message_id, and a short excerpt.",
        "default_prompt": "Use for temporal references like 'your first answer about X', 'what you said earlier regarding Y', 'when we started discussing Z'. Returns precise turn anchors so you can ground time-relative claims accurately. Combine with search_conversation_history to retrieve the full content of that turn.",
    },
    "search_web": {
        "id": "live_web_recon",
        "display_name": "Internet Search",
        "description": "Search the web for external, real-time, or post-training knowledge. Results are automatically cached in the thread knowledge base for future retrieval.",
        "default_prompt": "MANDATORY when web search is enabled: call search_web for virtually every factual question to supplement document content with current, external knowledge. Run it IN PARALLEL with document searches — do not wait for document results first. Never skip it based on pre-fetched document evidence alone. Always cite the URL and title of web results in your answer.",
    },
    "ask_for_clarification": {
        "id": "clarify_intent",
        "display_name": "Clarify Intent",
        "description": "Present the user with distinct interpretations of an ambiguous question.",
        "default_prompt": "Use only when the question has multiple plausible interpretations and making an assumption risks answering the wrong question entirely. Each option must be a complete, self-contained question.",
    },
    "get_thread_shape": {
        "id": "thread_shape",
        "display_name": "Thread Shape",
        "description": "Returns a snapshot of the thread's content inventory: document list with chunk counts and indexing status, plus QA history volume metrics.",
        "default_prompt": "Use to calibrate retrieval strategy: check document chunk counts to decide between search_documents vs. search_document_by_id, and check QA history volume to decide whether semantic memory search is worthwhile. Only call once — the snapshot is current at the time of the call.",
    },
}


class OrchestratorToolNode(ToolNode):
    async def ainvoke(self, input: dict, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        # Intercept tool calls to extract special JSON state updates
        res = await super().ainvoke(input, config, **kwargs)

        document_sources = list(input.get("document_sources", []))
        web_sources = list(input.get("web_sources", []))
        used_chat_ids = list(input.get("used_chat_ids", []))
        clarification_options = None

        messages = res.get("messages", [])
        for i, msg in enumerate(messages):
            if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and msg.content.startswith("{") and "__" in msg.content:
                try:
                    data = json.loads(msg.content)
                    # Replace the raw message content with the clean text
                    if "content" in data:
                        messages[i].content = data["content"]
                    if "__document_sources__" in data:
                        document_sources.extend(data["__document_sources__"])
                    if "__web_sources__" in data:
                        web_sources.extend(data["__web_sources__"])
                    if "__used_chat_ids__" in data:
                        used_chat_ids.extend(data["__used_chat_ids__"])
                    if "__clarification_options__" in data:
                        clarification_options = data["__clarification_options__"]
                        messages[i].content = f"Interrupted for clarification with options: {clarification_options}"
                except Exception as e:
                    logger.warning(f"Failed to parse tool JSON output: {e}")

        return {
            "messages": messages,
            "document_sources": document_sources,
            "web_sources": web_sources,
            "used_chat_ids": used_chat_ids,
            "clarification_options": clarification_options,
        }

async def call_model(state: AgentState, config: RunnableConfig):
    messages = state["messages"]
    llm = get_llm(state["llm_model"])
    iteration = state.get("iteration_count", 0) + 1
    
    context_window = state.get('context_window', DEFAULT_TOKEN_BUDGET)
    system_role = sanitize_system_role(state.get("system_role", ""))
    tool_instructions = normalize_tool_instructions(state.get("tool_instructions", {}))
    custom_instructions = sanitize_custom_instructions(state.get("custom_instructions", ""))
    reasoning_mode = state.get("reasoning_mode", True)
    reasoning_mode = state.get("reasoning_mode", True)

    use_web_search = state.get("use_web_search", False)
    intent_agent_ran = state.get("intent_agent_ran", True)
    reasoning_mode = state.get("reasoning_mode", True)
    prompt_content = build_system_prompt(
        context_window=context_window,
        system_role=system_role,
        tool_instructions=tool_instructions,
        custom_instructions=custom_instructions,
        use_web_search=use_web_search,
        intent_agent_ran=intent_agent_ran,
        reasoning_mode=reasoning_mode,
    )

    # Inject pre-fetched context bundle + pre-fetch-first retrieval policy
    bundle = state.get("pre_fetch_bundle")
    if bundle:
        bundle_text = _format_prefetch_for_prompt(bundle)
        if bundle_text:
            prompt_content += (
                "\n\nPRE-FETCH RETRIEVAL POLICY (LOCKED):\n"
                "Pre-fetched context (recent turns, semantic history, document evidence, document list) is\n"
                "already present in the PRE-FETCHED CONTEXT block below. Before calling any tool:\n"
                "1. Assess whether the pre-fetched content answers the rewritten query with confidence.\n"
                "   If YES, skip document/history tool calls — but NEVER skip search_web when it is available.\n"
                "2. If document evidence is present but the rewritten query is more specific than the raw question:\n"
                "   call search_documents or search_document_by_id ONCE with the rewritten query.\n"
                "3. If the question targets a specific document and its file_hash is in the document list:\n"
                "   prefer search_document_by_id (scoped) over search_documents (all documents).\n"
                "4. Do NOT call search_conversation_history just to re-read recent turns — the recent\n"
                "   conversation and semantic history are already in the pre-fetched block.\n"
                + (
                    "5. WEB SEARCH IS ENABLED AND MANDATORY: call search_web for this question IN PARALLEL\n"
                    "   with any document search. Pre-fetched document evidence does NOT replace a web search.\n"
                    "   Do not skip search_web regardless of how complete the pre-fetched content appears.\n"
                    if use_web_search else ""
                )
            ) + bundle_text

    prompt_template = build_chat_prompt()
    input_messages = prompt_template.format_messages(
        system_prompt=prompt_content,
        messages=messages,
    )

    llm_with_tools = llm.bind_tools(tools_list) if reasoning_mode else llm

    # Log complete prompt for Orchestrator Agent in OpenAI-like format
    logger.info(f"--- ORCHESTRATOR AGENT PROMPT BEGIN [thread_id: {state.get('thread_id')}, iteration: {iteration}] ---")
    payload = []
    for msg in input_messages:
        role = "system" if isinstance(msg, SystemMessage) else "user" if isinstance(msg, HumanMessage) else "assistant"
        entry = {"role": role, "content": msg.content}
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            entry["tool_calls"] = msg.tool_calls
        if isinstance(msg, ToolMessage):
            entry["role"] = "tool"
            entry["tool_call_id"] = msg.tool_call_id
        payload.append(entry)
    logger.info(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info(f"--- ORCHESTRATOR AGENT PROMPT END ---")

    response = await invoke_with_retry(llm_with_tools.ainvoke, input_messages)
    return {"messages": [response], "iteration_count": iteration}


def _looks_like_tool_call_text(text: str) -> bool:
    if not text:
        return False
    try:
        data = json.loads(text)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    if data.get("type") == "function":
        return True
    if "function" in data and "parameters" in data:
        return True
    if "tool" in data and "tool_input" in data:
        return True
    return False


async def force_final_answer(state: AgentState, config: RunnableConfig):
    """
    Fallback when the tool-iteration budget is exhausted or the model returns empty text.
    Rebuilds a clean, flat prompt from the retrieved tool content to avoid confusing the
    model with a broken tool-calling message chain (empty AIMessages, multiple ToolMessages).
    """
    messages = state["messages"]
    llm = get_llm(state["llm_model"])
    iteration = state.get("iteration_count", 0) + 1

    context_window = state.get('context_window', DEFAULT_TOKEN_BUDGET)
    system_role = sanitize_system_role(state.get("system_role", ""))
    tool_instructions = normalize_tool_instructions(state.get("tool_instructions", {}))
    custom_instructions = sanitize_custom_instructions(state.get("custom_instructions", ""))

    # ── Extract original user question ──
    original_question = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            original_question = msg.content if isinstance(msg.content, str) else str(msg.content)
            break  # take the first human message (the actual user question)

    # ── Collect all tool results and earlier clean AI turns ──
    tool_context_parts: list[str] = []
    prior_ai_parts: list[str] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content or ""
            # ToolMessages may still be raw JSON if OrchestratorToolNode didn't strip them
            if isinstance(content, str) and content.startswith("{") and "content" in content:
                try:
                    parsed = json.loads(content)
                    content = parsed.get("content", content)
                except Exception:
                    pass
            if content.strip():
                tool_context_parts.append(content.strip())
        elif isinstance(msg, AIMessage):
            txt = msg.content if isinstance(msg.content, str) else ""
            if isinstance(msg.content, list):
                from reasoning import _text_from_content_item
                txt = "\n".join([_text_from_content_item(i) for i in msg.content if i]).strip()
            # Only keep non-empty AI turns that are not tool-call-only turns
            if txt.strip() and not getattr(msg, "tool_calls", None):
                prior_ai_parts.append(txt.strip())

    # ── Build a direct synthesis prompt ──
    sys_prompt = build_system_prompt(
        context_window=context_window,
        system_role=system_role,
        tool_instructions=tool_instructions,
        custom_instructions=custom_instructions,
        reasoning_mode=reasoning_mode,
    )

    parts = []
    if tool_context_parts:
        parts.append("RETRIEVED CONTEXT (from tool searches):\n\n" + "\n\n---\n\n".join(tool_context_parts))
    if prior_ai_parts:
        parts.append("EARLIER ANALYSIS:\n\n" + "\n\n".join(prior_ai_parts))

    force_content = (
        "You MUST now write a final answer. Do NOT call any tools.\n\n"
        + ("\n\n".join(parts) + "\n\n" if parts else "")
        + f"USER QUESTION:\n{original_question}\n\n"
        "Write a complete, helpful answer based on the retrieved context above. "
        "Cite sources where available. If the context is insufficient, say so honestly."
    )
    force_msg = HumanMessage(content=force_content)

    prompt_template = build_chat_prompt()
    input_messages = prompt_template.format_messages(
        system_prompt=sys_prompt,
        messages=[force_msg],
    )
    response = await invoke_with_retry(llm.ainvoke, input_messages)
    return {"messages": [response], "iteration_count": iteration}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    reasoning_mode = state.get("reasoning_mode", True)

    if getattr(last_message, "tool_calls", None):
        if iteration_count >= max_iterations:
            # If the only pending call is search_web and no web search has run yet,
            # grant one extra pass so we don't force-finalize with zero web context.
            pending_tool_names = {tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "") for tc in last_message.tool_calls}
            web_sources_so_far = state.get("web_sources", [])
            if pending_tool_names == {"search_web"} and not web_sources_so_far:
                logger.info("Granting one extra iteration for search_web (no web results yet).")
                return "tools"
            logger.warning(f"Reaching max agent iterations ({iteration_count}/{max_iterations}). Forcing termination.")
            return "force_final_answer"
        return "tools"

    if not reasoning_mode and evidence_insufficient(state):
        if iteration_count >= max_iterations:
            logger.warning(f"Reaching max agent iterations ({iteration_count}/{max_iterations}). Forcing termination.")
            return "force_final_answer"
        logger.info("Non-reasoning mode: auto-tools pass triggered (no tool calls, insufficient evidence).")
        return "auto_tools"

    # Detect empty-content response after tool execution (e.g. model outputs nothing after
    # receiving tool results). Force a final answer instead of silently ending with blank text.
    if iteration_count > 0:
        content = getattr(last_message, "content", "")
        if isinstance(content, list):
            from reasoning import _text_from_content_item
            text_body = "\n".join([_text_from_content_item(i) for i in content if i]).strip()
        else:
            text_body = (content or "").strip()

        if not text_body:
            logger.warning("LLM returned empty response after tool execution. Triggering force_final_answer.")
            return "force_final_answer"

    if not reasoning_mode:
        content = getattr(last_message, "content", "")
        if isinstance(content, list):
            from reasoning import _text_from_content_item
            content = "\n".join([_text_from_content_item(i) for i in content if i]).strip()
        if isinstance(content, str) and _looks_like_tool_call_text(content.strip()):
            logger.warning("Non-reasoning mode: model returned tool-call-like JSON. Forcing final answer.")
            return "force_final_answer"

    return END


async def auto_tools(state: AgentState, config: RunnableConfig):
    """
    Non-reasoning fallback: run required tools when the model fails to call any.
    """
    tool_messages: list[ToolMessage] = []
    document_sources: list[Dict[str, Any]] = []
    web_sources: list[Dict[str, Any]] = []
    used_chat_ids: list[str] = []

    working_query = state.get("working_query", "")
    use_web_search = state.get("use_web_search", False)
    intent_ref = state.get("intent_reference_type", "NONE")
    prefetch = state.get("pre_fetch_bundle") or {}
    documents = prefetch.get("documents") or []

    async def _run_tool(tool_name: str, result: str):
        tool_messages.append(ToolMessage(content=result, tool_call_id=f"auto_{tool_name}"))
        collect_tool_sources(result, document_sources, web_sources, used_chat_ids)

    if use_web_search and not state.get("web_sources"):
        logger.info("Auto-tools: running search_web.")
        result = await search_web(working_query, config=config)
        await _run_tool("search_web", result)

    if documents and not state.get("document_sources"):
        if intent_ref == "ENTITY" and len(documents) == 1:
            file_hash = documents[0].get("file_hash")
            if file_hash:
                logger.info("Auto-tools: running search_document_by_id.")
                result = await search_document_by_id(working_query, file_hash, config=config)
                await _run_tool("search_document_by_id", result)
        else:
            logger.info("Auto-tools: running search_documents.")
            result = await search_documents(working_query, config=config)
            await _run_tool("search_documents", result)

    if intent_ref == "SEMANTIC":
        logger.info("Auto-tools: running search_conversation_history.")
        result = await search_conversation_history(working_query, config=config)
        await _run_tool("search_conversation_history", result)

    return {
        "messages": tool_messages,
        "document_sources": document_sources,
        "web_sources": web_sources,
        "used_chat_ids": used_chat_ids,
    }

def clarification_router(state: AgentState):
    if state.get("clarification_options"):
        return END  # Suspend graph
    return "agent"


# Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", OrchestratorToolNode(tools_list))
workflow.add_node("force_final_answer", force_final_answer)
workflow.add_node("auto_tools", auto_tools)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "auto_tools": "auto_tools", "force_final_answer": "force_final_answer", END: END},
)
workflow.add_edge("auto_tools", "agent")

workflow.add_conditional_edges("tools", clarification_router, {END: END, "agent": "agent"})
workflow.add_edge("force_final_answer", END)

app = workflow.compile()


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
    use_web_search: bool = False,
    intent_agent_ran: bool = True,
    reasoning_mode: bool = True,
) -> str:
    """Build the Orchestrator Agent system prompt."""
    role = system_role or "Expert AI Research Assistant specializing in analyzing uploaded documents and synthesizing accurate answers."
    catalog = get_tool_catalog()
    playbook = normalize_tool_instructions(tool_instructions or {})
    
    # Load base template
    template = get_orchestrator_prompt() if reasoning_mode else get_orchestrator_prompt_compact()
    
    # Setup variables for template substitution
    if intent_agent_ran:
        intent_agent_note = "The Intent Agent upstream has already rewritten the user's query and classified its coverage; your job is to:"
        preprocessing_phase_note = ""
        phase0 = ""
        phase_count = "five"
        phase_start = ""
        orient_word = "rewritten"
        orient_extra = ""
        plan_query_note = ""
    else:
        intent_agent_note = "No upstream query preprocessor ran for this turn — you are responsible for both query preprocessing AND orchestration. Your job is to:"
        preprocessing_phase_note = "  0. Preprocess the raw user query: resolve coreferences, standalone-ify, assess coverage (Phase 0)."
        phase0 = get_orchestrator_phase0_prompt() if reasoning_mode else get_orchestrator_phase0_prompt_compact()
        phase_count = "six"
        phase_start = " Begin with Phase 0 — Preprocess."
        orient_word = "working"
        orient_extra = "\n  e) Does the raw message contain unresolved pronouns or references? → your Phase 0\n     WORKING QUERY replaces the raw message for all retrieval operations below."
        plan_query_note = "\n  - Use the WORKING QUERY from Phase 0 — not the raw user message — for all tool arguments."
    
    max_parallel_tools = 4 if use_web_search else 3
    
    # Build tool registry/playbook sections
    EDIT = "(USER-CONFIGURABLE)"
    tool_registry_section = (
        f"\n\n{'=' * 64}\nTOOL REGISTRY {EDIT}:\n{'=' * 64}\n"
        + "\n".join(
            [
                f"- {item['display_name']} (tool name: `{item['tool_name']}`)\n    {item['description']}"
                for item in catalog
            ]
        )
    )
    tool_playbook_section = (
        f"\n\n{'=' * 64}\nTOOL PLAYBOOK {EDIT}:\n{'=' * 64}\n"
        + "\n".join(
            [
                f"- `{item['tool_name']}`: {playbook.get(item['id'], item['default_prompt'])}"
                for item in catalog
            ]
        )
    )

    # Build web search mandate section if enabled
    web_search_mandate_section = ""
    if use_web_search:
        LOCK = "(LOCKED — not overridable)"
        web_search_mandate_section = (
            f"\n\n{'=' * 64}\nWEB SEARCH MANDATE {LOCK} — overrides pre-fetch sufficiency\n"
            f"{'=' * 64}\n"
            + get_web_search_mandate()
        )

    # Build custom instructions section if provided
    custom_instructions_section = ""
    if custom_instructions:
        custom_instructions_section = (
            f"\n\n{'=' * 64}\nUSER CUSTOM INSTRUCTIONS {EDIT}\n{'=' * 64}\n"
            + custom_instructions
        )

    # Substitute placeholders in template
    prompt = template.format(
        SYSTEM_ROLE=role,
        CONTEXT_WINDOW=context_window,
        INTENT_AGENT_NOTE=intent_agent_note,
        PREPROCESSING_PHASE_NOTE=preprocessing_phase_note,
        PREPROCESSING_SECTION=phase0,
        PHASE_COUNT=phase_count,
        PHASE_START=phase_start,
        ORIENT_WORD=orient_word,
        ORIENT_EXTRA=orient_extra,
        PLAN_QUERY_NOTE=plan_query_note,
        MAX_PARALLEL_TOOLS=max_parallel_tools,
        TOOL_REGISTRY_SECTION=tool_registry_section,
        TOOL_PLAYBOOK_SECTION=tool_playbook_section,
        WEB_SEARCH_MANDATE_SECTION=web_search_mandate_section,
        CUSTOM_INSTRUCTIONS_SECTION=custom_instructions_section,
    )

    return prompt
