import asyncio
import logging
import json
from typing import TypedDict, List, Annotated, Dict, Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from models import get_llm, get_embedding_model, DEFAULT_TOKEN_BUDGET, DEFAULT_MAX_ITERATIONS
from vectordb.qdrant import QdrantAdapter

logger = logging.getLogger(__name__)

search_tool = DuckDuckGoSearchResults(output_format="list", num_results=6)

async def invoke_with_retry(func, *args, **kwargs):
    max_retries = 10
    delay = 5
    for i in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            # 503 – model still loading / service unavailable
            is_503_loading = "503" in err_str and ("loading" in err_str or "unavailable" in err_str)
            # 400 – LM Studio unloaded the model between requests
            is_model_unloaded = "400" in err_str and "model unloaded" in err_str
            # 429 – rate limit / too many requests
            is_rate_limit = "429" in err_str or "rate limit" in err_str or "too many requests" in err_str

            if is_503_loading or is_model_unloaded or is_rate_limit:
                reason = (
                    "Model is loading (503)" if is_503_loading
                    else "Model was unloaded (400)" if is_model_unloaded
                    else "Rate limited (429)"
                )
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
    pdf_sources: List[Dict[str, Any]]
    web_sources: List[Dict[str, Any]]
    used_chat_ids: List[str]
    clarification_options: Optional[List[str]]
    iteration_count: int
    max_iterations: int
    system_role: str
    tool_instructions: Dict[str, str]
    custom_instructions: str
    pre_fetch_bundle: Optional[Dict[str, Any]]


@tool
async def search_documents(query: str, max_results: int = 10, config: RunnableConfig = None) -> str:
    """
    Perform a semantic vector search over all uploaded PDF documents AND previously cached
    internet search results for this thread.
    Returns the most relevant chunks along with neighboring context passages for continuity.

    This uses embedding-based similarity — phrase queries as natural questions rather than
    keyword strings for best results. If the first call returns weak or irrelevant evidence,
    retry with a rephrased or more specific query before concluding the information is absent.

    Each returned passage is prefixed with its source so you can cite it accurately:
      - PDF passages: [Source: Document "<filename>"]
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

        db = QdrantAdapter()

        # ── Build a file_hash → file_name lookup from the thread's document list ──
        from database import get_thread_files
        try:
            thread_files = await get_thread_files(thread_id)
            hash_to_name = {f.file_hash: f.file_name for f in thread_files}
        except Exception:
            hash_to_name = {}

        # ── PDF chunk search with neighbor expansion ──
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

        # ── Cached web search results ──
        web_chunks = await db.search_web_chunks(
            thread_id=thread_id,
            query_vector=query_vector,
            limit=max(3, max_results // 3),
        )

        if not expanded_pdf_chunks and not web_chunks:
            return "No relevant content found in documents or cached web results."

        pdf_sources = []
        web_sources = []
        context_parts = []

        for chunk in expanded_pdf_chunks:
            text = chunk.get("text", "")
            fh = chunk.get("file_hash", "")
            fname = hash_to_name.get(fh, fh)
            labeled = f'[Source: Document "{fname}"]\n{text}'
            context_parts.append(labeled)
            pdf_sources.append({
                "text": text[:200] + "..." if len(text) > 200 else text,
                "file_hash": fh,
                "file_name": fname,
                "score": chunk.get("score", 0.0),
            })

        for wchunk in web_chunks:
            text = wchunk.get("text", "")
            url = wchunk.get("url", "")
            title = wchunk.get("title", url)
            labeled = f'[Source: Internet Search — "{title}" | {url}]\n{text}'
            context_parts.append(labeled)
            web_sources.append({
                "text": text[:200] + "..." if len(text) > 200 else text,
                "url": url,
                "title": title,
                "score": wchunk.get("score", 0.0),
            })

        result: Dict[str, Any] = {"content": "\n\n".join(context_parts)}
        if pdf_sources:
            result["__pdf_sources__"] = pdf_sources
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
async def search_web(query: str, config: RunnableConfig = None) -> str:
    """
    Search the web for external, real-time, or post-training knowledge.

    Results are automatically stored in the thread's knowledge base so that future
    questions on the same topic can be answered without a new web request.

    Use this along with search_documents when you need to augment the knowledge from
    uploaded documents with the latest information from the internet. This helps
    provide a more comprehensive answer by checking both internal PDFs and external
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

        raw = await asyncio.to_thread(search_tool.invoke, query)
        logger.info(f"Raw search results for '{query}': {raw}")

        # DuckDuckGoSearchResults returns a list of dicts; fall back to string if not.
        if isinstance(raw, list):
            results_list = raw
        elif isinstance(raw, str):
            try:
                results_list = json.loads(raw)
            except Exception:
                # Plain-text fallback — wrap as a single result without URL
                results_list = [{"snippet": raw, "title": query, "link": ""}]
        else:
            results_list = []

        if not results_list:
            return "Web search returned no results."

        texts = [r.get("snippet", r.get("body", "")) for r in results_list]
        urls  = [r.get("link",    r.get("href",  "")) for r in results_list]
        titles = [r.get("title", "") for r in results_list]

        # Remove empty snippets
        valid = [(t, u, ti) for t, u, ti in zip(texts, urls, titles) if t.strip()]
        if not valid:
            return "Web search returned no usable text."
        texts, urls, titles = zip(*valid)
        texts  = list(texts)
        urls   = list(urls)
        titles = list(titles)

        # ── Persist results in Qdrant for future retrieval ──
        if thread_id and embedding_model:
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

        # ── Build source-labeled content for the LLM ──
        context_parts = []
        web_sources = []
        for text, url, title in zip(texts, urls, titles):
            label = title or url or "Internet Search"
            labeled = f'[Source: Internet Search — "{label}" | {url}]\n{text}'
            context_parts.append(labeled)
            web_sources.append({
                "text": text[:200] + "..." if len(text) > 200 else text,
                "url": url,
                "title": title,
            })

        return json.dumps({
            "content": "\n\n".join(context_parts),
            "__web_sources__": web_sources,
        })
    except Exception as e:
        logger.error(f"Web search failed: {e}", exc_info=True)
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

        # Resolve file name for source attribution
        from database import get_thread_files
        try:
            thread_files = await get_thread_files(thread_id)
            hash_to_name = {f.file_hash: f.file_name for f in thread_files}
            fname = hash_to_name.get(file_hash, file_hash)
        except Exception:
            fname = file_hash

        sources = []
        context_parts = []
        for chunk in expanded_chunks:
            text = chunk.get("text", "")
            labeled = f'[Source: Document "{fname}"]\n{text}'
            context_parts.append(labeled)
            sources.append({
                "text": text[:200] + "..." if len(text) > 200 else text,
                "file_hash": file_hash,
                "file_name": fname,
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


async def call_intent_model(state: IntentAgentState, config: RunnableConfig):
    """
    Single-pass Intent Agent: classifies and rewrites the user's query for the Orchestrator.
    No tools, no retries — one LLM call, one JSON output.
    """
    messages = state["messages"]
    llm = get_llm(state["llm_model"], temperature=0.0)
    context_window = state.get("context_window", DEFAULT_TOKEN_BUDGET)

    system_prompt = (
        f"""You are the Query Preprocessor — a single-pass retrieval optimizer that runs before the Orchestrator Agent.
        Your output is a JSON routing signal consumed by the Orchestrator; it is NEVER shown to the user.

        YOUR ROLE (production RAG pattern):
        This is the "Rewrite-Retrieve-Read" step (arXiv:2305.14283) combined with Standalone Question
        Generation. You transform the user's raw message into an optimal search query before it is
        embedded as a vector for semantic similarity search.
        The rewritten_query you produce is embedded once and searched across all retrieval tools.

        WHY MINIMAL, FAITHFUL REWRITING MATTERS:
        The rewritten_query is passed directly to a cosine-similarity vector search.
        Adding topics, subtopics, or angles the user never mentioned DILUTES the query embedding —
        it shifts the vector away from the user's actual intent and returns chunks ranked for the
        wrong topics, burying the real answer. Your job is coreference resolution + standalone-ification,
        NOT academic elaboration or question enhancement.

        ORCHESTRATOR'S TOOLS (your rewrite optimizes for all of these):
        - search_documents       → semantic search across ALL uploaded PDFs + cached web chunks
        - search_pdf_by_document → same, scoped to a single file by hash (for named-document questions)
        - search_conversation_history → semantic search over all prior Q&A pairs in this thread
        - search_web             → live web search; results cached back into the vector store
        - ask_for_clarification  → surfaces disambiguation options to the user (genuine ambiguity only)

        CONTEXT AVAILABLE NOW:
        - Conversation history: in the messages below (oldest → newest)
        - Pre-fetched bundle: recent turns, semantic memory, PDF evidence, document list — appended below

        RUNTIME: {context_window} tokens. No tool calls. No preamble. Respond with JSON only.

        ════════════════════════════════════════════════════
        OUTPUT — valid JSON only, no markdown fences, no extra text:
        {{
          "status": "CLEAR_STANDALONE" | "CLEAR_FOLLOWUP" | "AMBIGUOUS",
          "rewritten_query": "<the optimized search query>",
          "reference_type": "NONE" | "SEMANTIC" | "TEMPORAL" | "ENTITY",
          "context_coverage": "SUFFICIENT" | "PROBABLY_SUFFICIENT" | "INSUFFICIENT",
          "clarification_options": ["Full question A", "Full question B"] | null
        }}
        ════════════════════════════════════════════════════

        REWRITING ALGORITHM — apply these steps in order:

        STEP 1 — COREFERENCE RESOLUTION (critical for follow-ups)
        Replace every pronoun, "it", "this", "that", "they", "the method", "the document" etc.
        with its explicit referent from the conversation history.
          "How does it work?" (after discussing BERT) → "How does BERT work?"
          "What were the main findings?" (after uploading a paper) → "What are the main findings in [paper title or 'the uploaded document']?"
          "Tell me more about that" (after discussing attention mechanisms) → "Explain attention mechanisms in more detail"

        STEP 2 — STANDALONE-IFY (minimum context for cold retrieval)
        Add only enough subject/domain context so a cold vector search with no conversation history
        would retrieve the right chunks. Nothing more.
          "What variants exist?" (after discussing glioblastoma) → "What variants of glioblastoma have been discovered?"
          NOT: "What variants of glioblastoma exist, including subtypes, IDH mutations, and WHO classification?"

        STEP 3 — PRESERVE SCOPE EXACTLY
        The user's question has a scope. Do NOT widen or narrow it.

        FORBIDDEN — SCOPE WIDENING (adding topics the user never mentioned):
          "what is glioblastoma, are there new variants?" → "What are the characteristics, causes, and new variants of glioblastoma?"
          ↑ BAD: "characteristics" and "causes" were never asked for

          "Explain transformers" → "Explain the transformer architecture including self-attention, positional encoding, and feed-forward layers"
          ↑ BAD: user asked to explain it, not enumerate every sub-component

        FORBIDDEN — SCOPE NARROWING (collapsing a broad question to one aspect):
          "What are the main findings?" → "What are the statistical findings in the results section?"
          ↑ BAD: user didn't specify results section or statistics

        CORRECT:
          "what is glioblastoma, are there new variants?" → "What is glioblastoma, and have any new variants been discovered?"
          "Explain transformers" → "Explain the transformer architecture in machine learning"
          "What are the main findings?" → "What are the main findings or conclusions in the uploaded document?"
          "How does RAG work?" → "How does Retrieval-Augmented Generation (RAG) work?"
          "Summarize it" (after PDF upload) → "Provide a summary of the uploaded document"

        STEP 4 — ONE CLEAN QUESTION
        Output a single natural question. No "Q:" prefix, no bullet lists, no semicolon-joined sub-questions.
        If the user asked multiple related things (as in glioblastoma above), preserve them as a single
        compound question using natural conjunctions.

        ════════════════════════════════════════════════════
        CLASSIFICATION:

        CLEAR_STANDALONE — Self-contained, no prior-context references. All first messages in a thread are this.
        CLEAR_FOLLOWUP   — References prior context, but the referent is unambiguously resolved from history.
        AMBIGUOUS        — Multiple genuinely different interpretations AND history cannot resolve which one.
                           HIGH BAR: Only use this if guessing would answer an entirely different question.
                           "Tell me more" is NOT ambiguous — it continues the last topic.
                           A pronoun with one clear referent is NOT ambiguous.

        CONTEXT_COVERAGE — controls how many tool calls the Orchestrator is budgeted:
        SUFFICIENT          — Pre-fetched bundle directly and fully answers the rewritten query.
                              The Orchestrator should synthesize from pre-fetch, no extra retrieval needed.
                              Use for: well-known factual questions, greetings, questions fully answered in pre-fetch.
        PROBABLY_SUFFICIENT — Pre-fetched bundle has partial or adjacent content. One targeted tool call may sharpen the answer.
        INSUFFICIENT        — Pre-fetched content clearly lacks what's needed. Orchestrator must retrieve.
                              Use for: first messages (nothing pre-fetched), deep history questions,
                              questions about specific doc sections not visible in pre-fetch.

        REFERENCE_TYPE:
          NONE     — No dependency on prior conversation
          SEMANTIC — References a topic/concept discussed earlier ("what did we say about X?")
          TEMPORAL — References a chronological position ("your first answer", "earlier you mentioned")
          ENTITY   — References a specific named thing from earlier ("that figure", "the equation", "that method")"""
    )

    # Inject pre-fetched context bundle
    bundle = state.get("pre_fetch_bundle")
    if bundle:
        system_prompt += _format_prefetch_for_prompt(bundle)

    if not messages or not isinstance(messages[0], SystemMessage):
        input_messages = [SystemMessage(content=system_prompt)] + messages
    else:
        input_messages = [SystemMessage(content=system_prompt)] + messages[1:]

    # Single direct call — no tools, no retries
    try:
        response = await invoke_with_retry(llm.ainvoke, input_messages)
    except Exception as e:
        logger.error(f"Intent Agent LLM call failed: {e}")
        original_question = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        intent_result = {
            "status": "CLEAR_STANDALONE",
            "rewritten_query": original_question,
            "reference_type": "NONE",
            "context_coverage": "PROBABLY_SUFFICIENT",
            "clarification_options": None,
        }
        return {"messages": [AIMessage(content=json.dumps(intent_result))], "iteration_count": 1, "intent_result": intent_result}

    # Parse the JSON response
    intent_result = None
    content = response.content.strip()
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        intent_result = json.loads(content)
    except Exception as e:
        logger.error(f"Failed to parse intent JSON: {e}")
        original_question = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        intent_result = {
            "status": "CLEAR_STANDALONE",
            "rewritten_query": original_question,
            "reference_type": "NONE",
            "context_coverage": "PROBABLY_SUFFICIENT",
            "clarification_options": None,
        }

    return {
        "messages": [response],
        "iteration_count": 1,
        "intent_result": intent_result,
    }


intent_workflow = StateGraph(IntentAgentState)
intent_workflow.add_node("agent", call_intent_model)
intent_workflow.add_edge(START, "agent")
intent_workflow.add_edge("agent", END)

intent_app = intent_workflow.compile()
# --- End Intent Agent ---

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
        "description": "Search the web for external, real-time, or post-training knowledge. Results are automatically cached in the thread knowledge base for future retrieval.",
        "default_prompt": "MANDATORY when web search is enabled: call search_web for virtually every factual question to supplement PDF content with current, external knowledge. Run it IN PARALLEL with document searches — do not wait for PDF results first. Never skip it based on pre-fetched PDF evidence alone. Always cite the URL and title of web results in your answer.",
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

        pdf_sources = list(input.get("pdf_sources", []))
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
                    if "__pdf_sources__" in data:
                        pdf_sources.extend(data["__pdf_sources__"])
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
            "pdf_sources": pdf_sources,
            "web_sources": web_sources,
            "used_chat_ids": used_chat_ids,
            "clarification_options": clarification_options,
        }

async def call_model(state: AgentState, config: RunnableConfig):
    messages = state["messages"]
    llm = get_llm(state["llm_model"])
    iteration = state.get("iteration_count", 0) + 1
    
    llm_with_tools = llm.bind_tools(tools_list)
    
    context_window = state.get('context_window', DEFAULT_TOKEN_BUDGET)
    system_role = sanitize_system_role(state.get("system_role", ""))
    tool_instructions = normalize_tool_instructions(state.get("tool_instructions", {}))
    custom_instructions = sanitize_custom_instructions(state.get("custom_instructions", ""))

    use_web_search = state.get("use_web_search", False)
    prompt_content = build_system_prompt(
        context_window=context_window,
        system_role=system_role,
        tool_instructions=tool_instructions,
        custom_instructions=custom_instructions,
        use_web_search=use_web_search,
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
                "   If YES, skip PDF/history tool calls — but NEVER skip search_web when it is available.\n"
                "2. If PDF evidence is present but the rewritten query is more specific than the raw question:\n"
                "   call search_documents or search_pdf_by_document ONCE with the rewritten query.\n"
                "3. If the question targets a specific document and its file_hash is in the document list:\n"
                "   prefer search_pdf_by_document (scoped) over search_documents (all documents).\n"
                "4. Do NOT call search_conversation_history just to re-read recent turns — the recent\n"
                "   conversation and semantic history are already in the pre-fetched block.\n"
                + (
                    "5. WEB SEARCH IS ENABLED AND MANDATORY: call search_web for this question IN PARALLEL\n"
                    "   with any document search. Pre-fetched PDF evidence does NOT replace a web search.\n"
                    "   Do not skip search_web regardless of how complete the pre-fetched content appears.\n"
                    if use_web_search else ""
                )
            ) + bundle_text

    sys_prompt = SystemMessage(content=prompt_content)

    # Langchain expects SystemMessage at the start
    input_messages = [sys_prompt] + messages
    response = await invoke_with_retry(llm_with_tools.ainvoke, input_messages)
    return {"messages": [response], "iteration_count": iteration}


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
    sys_prompt = SystemMessage(content=build_system_prompt(
        context_window=context_window,
        system_role=system_role,
        tool_instructions=tool_instructions,
        custom_instructions=custom_instructions
    ))

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

    response = await invoke_with_retry(llm.ainvoke, [sys_prompt, force_msg])
    return {"messages": [response], "iteration_count": iteration}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)

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
) -> str:
    role = system_role or "Expert AI Research Assistant specializing in analyzing uploaded documents and synthesizing accurate answers."
    catalog = get_tool_catalog()
    playbook = normalize_tool_instructions(tool_instructions or {})

    # ── Section helpers ──────────────────────────────────────────────────────
    LOCK  = "(LOCKED — not overridable)"
    EDIT  = "(USER-CONFIGURABLE)"
    sep   = "─" * 60

    sections = [
        # ────────────────────────────────────────────────────────────────────
        (
            f"IDENTITY & MISSION {EDIT}",
            f"""You are {role}

Your architectural role is the Orchestrator in a production Retrieval-Augmented Generation (RAG)
system. The Intent Agent upstream has already rewritten the user's query and classified its
coverage; your job is to:
  1. Orient on what is already known (pre-fetched context).
  2. Plan which tools to call — and in what parallel groupings — to fill evidence gaps.
  3. Retrieve evidence by dispatching tools.
  4. Assess quality: decide whether evidence is sufficient or a retry is warranted.
  5. Synthesize a grounded, well-cited final answer."""
        ),

        # ────────────────────────────────────────────────────────────────────
        (
            f"RUNTIME CONSTRAINTS {LOCK}",
            f"""Context window: {context_window} tokens. You share this budget with the conversation history,
pre-fetched context, tool results, and your final answer. Manage it actively:
  • Prefer targeted queries over broad ones to keep tool results concise.
  • Do not repeat identical or near-identical tool calls — variation must be meaningful.
  • If iteration budget is low (visible from iteration_count nearing max_iterations), skip
    optional confirmatory searches and move directly to synthesis."""
        ),

        # ────────────────────────────────────────────────────────────────────
        (
            f"REASONING PROTOCOL {LOCK}",
            f"""Execute every response in these five ordered phases. Do NOT skip phases.

{sep}
PHASE 1 — ORIENT  (silent, no output)
{sep}
Read the PRE-FETCHED CONTEXT block (if present). Ask yourself:
  a) Does it directly answer the rewritten question with enough specificity?
  b) Are there named documents in the document list that are clearly relevant?
  c) Is the question asking about something that happened after the documents were written
     (current events, real-time data)? → web search may be mandatory.
  d) Does the question reference a prior exchange? → semantic history may be needed.
Record your answers internally; they drive Phase 2.

{sep}
PHASE 2 — PLAN  (concise, visible: 1-3 lines)
{sep}
Output a brief retrieval plan before calling any tools, e.g.:
  "Calling search_documents with [query] and search_web with [query] in parallel."
  "Pre-fetch content is sufficient for this factual question — no extra retrieval needed."
  "Will call search_pdf_by_document scoped to [filename] (hash known from document list)."

Rules:
  • Group every independent tool call into a SINGLE parallel batch — dispatch them together.
  • Never call tool B only after tool A returns if they are independent of each other.
  • Avoid calling more than {3 if not use_web_search else 4} tools in one batch (context budget).
  • State what you expect each tool to return — this prevents redundant follow-up calls.

{sep}
PHASE 3 — RETRIEVE  (tool calls)
{sep}
Execute the plan from Phase 2. Apply these dispatch rules:
  PARALLEL FIRST — independent searches MUST be batched together in one response turn,
  never issued sequentially.

  TOOL SELECTION DECISION TREE:
  ┌─ Is the question answerable from pre-fetched context alone?
  │    YES → set context_coverage = SUFFICIENT; skip all retrieval tools except search_web
  │    NO  ↓
  ├─ Does the question name a specific document?
  │    YES → use search_pdf_by_document (scoped, avoids noise from other files)
  │    NO  → use search_documents (all-document semantic search)
  │
  ├─ Does the question reference a past topic discussed in this thread?
  │    YES → use search_conversation_history IN PARALLEL with the document search
  │    NO  → skip search_conversation_history (recent history is in pre-fetch)
  │
  ├─ Is web search enabled AND is this a factual/informational question?
  │    YES → call search_web IN PARALLEL with document searches — MANDATORY, not optional
  │    NO  → skip search_web
  │
  └─ Is the question genuinely ambiguous with multiple distinct interpretations?
       YES → call ask_for_clarification — stop all other retrieval
       NO  → never call ask_for_clarification

  RETRY LOGIC — if a tool returns insufficient or off-topic results:
    1st retry: rephrase the query (more specific, different vocabulary, drop stop words).
    2nd retry: decompose the question and search for sub-components separately.
    After 2 retries with no improvement: accept partial evidence and note the gap in synthesis.

{sep}
PHASE 4 — ASSESS  (silent self-check)
{sep}
Before writing a single word of the final answer, evaluate:
  ✓ Coverage: Does retrieved evidence directly address the user's question?
  ✓ Confidence: Are the key claims backed by at least one retrievable source?
  ✓ Conflicts: Do sources contradict each other? → must be flagged in the answer.
  ✓ Gaps: Is there a material gap in evidence? → either retry (if budget remains) or
          disclose the gap explicitly in the answer; never fill gaps with guesses.

SUFFICIENCY CRITERIA — stop retrieving when ANY of these is true:
  • Retrieved passages directly answer the question with specific supporting detail.
  • Two independent tool calls with varied queries both return the same result (convergence).
  • Iteration count has reached max_iterations — synthesize from whatever is available.
  • Query is a greeting, meta-question, or does not require factual retrieval.

INSUFFICIENCY SIGNALS — retrieve more when ALL of these are true:
  • No passage contains the specific fact the question asks for.
  • A rephrased query has not been tried yet.
  • Iteration budget has not been exhausted.

{sep}
PHASE 5 — SYNTHESIZE  (final answer)
{sep}
Write the final answer only after completing Phase 4. Quality bar:
  • Lead with the most directly relevant finding — do not bury the answer in preamble.
  • Integrate evidence from multiple sources into a unified narrative; do not dump chunks.
  • Use Markdown formatting (headers, bullets, bold) when the answer is multi-part or complex.
  • Match answer depth to question complexity: short factual questions get concise answers;
    analytical questions get structured, multi-paragraph answers.
  • Proactively note uncertainty, limitations, or conflicting evidence.
  • NEVER output a final answer that consists solely of tool output verbatim — always
    add synthesis, context, and explanation."""
        ),

        # ────────────────────────────────────────────────────────────────────
        (
            f"TOOL REGISTRY {EDIT}",
            "\n".join([
                f"- {item['display_name']} (tool name: `{item['tool_name']}`)\n"
                f"    {item['description']}"
                for item in catalog
            ])
        ),

        # ────────────────────────────────────────────────────────────────────
        (
            f"TOOL PLAYBOOK {EDIT}",
            "\n".join([
                f"- `{item['tool_name']}`: {playbook.get(item['id'], item['default_prompt'])}"
                for item in catalog
            ])
        ),

        # ────────────────────────────────────────────────────────────────────
        (
            f"CITATION STANDARDS {LOCK}",
            """Every factual claim in your final answer MUST be traceable to a source. Apply these rules:

  PDF documents:
    Inline: 'According to [filename], ...' or '([filename], p. N if page-numbered)'
    When multiple PDFs corroborate: 'Both [file-a] and [file-b] state that ...'
    Never invent filenames — use only names returned by search tools or list_uploaded_documents.

  Internet search results:
    Inline: 'According to [Page Title] (source: <url>), ...'
    Always include both title and URL if both are available in the search result.
    Never cite a web result without a URL — if the URL is missing, say 'a web source found that'.

  Conversation history / semantic memory:
    Inline: 'As we discussed earlier, ...' or 'Based on a prior exchange in this thread, ...'

  Internal knowledge (no retrieved source):
    Clearly mark: 'Based on general knowledge (not from your documents), ...'
    Use sparingly — prefer retrieved evidence over internal knowledge for factual claims.

  Conflicting sources:
    'According to [source-A], X; however, [source-B] states Y — these sources disagree.'
    Do NOT silently pick one side; surface the disagreement.

  Evidence gaps:
    'The uploaded documents do not contain specific information about X.'
    'A web search did not return relevant results for this query.'
    Never fabricate a citation or fill a gap with plausible-sounding but unchecked facts."""
        ),

        # ────────────────────────────────────────────────────────────────────
        (
            f"ANTI-PATTERNS {LOCK}",
            """Avoid these failure modes that degrade answer quality:

  ✗ Answering before retrieving  — do not synthesize from internal knowledge when tools
      would return better evidence. Tools exist for a reason; use them first.

  ✗ Skipping parallel execution  — calling search_documents and then search_web in
      separate sequential turns when they are independent wastes the iteration budget.

  ✗ Redundant tool calls  — retrying with an identical or trivially paraphrased query
      wastes tokens without improving evidence quality. Materially rephrase or decompose.

  ✗ Pre-fetch over-reliance  — treating pre-fetched PDF evidence as equivalent to a
      freshly targeted search when the rewritten query is more specific than the raw question.
      When the rewritten query is significantly more specific, re-query with the precise terms.

  ✗ Evidence laundering  — presenting internal knowledge as if it came from a tool result.
      Only cite sources that actually appeared in tool output.

  ✗ Premature clarification  — asking the user to clarify when the question's intent is
      recoverable from conversation history. ask_for_clarification is a last resort.

  ✗ Verbatim chunk dumping  — pasting raw retrieved passages as the final answer without
      synthesis. Always transform evidence into a coherent, user-facing response.

  ✗ Ignoring conflicts  — blending contradictory claims from different sources into a
      single coherent-sounding statement that misrepresents both sources.

  ✗ Fabricating detail to fill gaps  — if evidence is absent, say so. Never invent
      specific facts, statistics, dates, or names that were not returned by tools."""
        ),
    ]

    # ── Conditional: web search mandate ─────────────────────────────────────
    if use_web_search:
        sections.append((
            f"WEB SEARCH MANDATE {LOCK} — overrides pre-fetch sufficiency",
            f"""Internet Search (search_web) is ENABLED for this session.

MANDATORY INVOCATION — call search_web for every factual or informational question:
  • Run search_web IN PARALLEL with search_documents / search_pdf_by_document in Phase 3.
  • Pre-fetched PDF evidence does NOT satisfy this mandate — PDF and web are complementary.
  • Do not defer web search to a second iteration after checking PDF results — batch them.

SOLE EXCEPTIONS (the only cases where search_web may be skipped):
  • Pure conversation meta-questions: 'how many messages have we had?', 'can you summarize our chat?'
  • The user's question is entirely answered by their own just-provided context (e.g., 'fix this text I pasted').
  • Clarification exchanges where no factual retrieval is needed.

When query rephrasing is needed for web search, use a concise keyword-rich query rather
than a full natural-language question — web search engines rank keyword density differently
from embedding-based vector search."""
        ))

    # ── Conditional: custom user instructions ────────────────────────────────
    if custom_instructions:
        sections.append((f"USER CUSTOM INSTRUCTIONS {EDIT}", custom_instructions))

    return "\n\n".join([f"{'═' * 64}\n{title}:\n{'═' * 64}\n{body}" for title, body in sections])
