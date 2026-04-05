"""
rag.py - Document indexing for RAG Service

This module handles:
- Text chunking and embedding generation
- Per-thread collection indexing
- PDF download and parsing
"""

import os
import logging
import asyncio
from typing import Dict, Any, List, Optional

import httpx
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.llm_server_client import (
    get_embedding_model, get_llm, 
    DEFAULT_TOKEN_BUDGET, RATIO_MEMORY_SUMMARIZATION_THRESHOLD, 
    RATIO_MEMORY_HARD_LIMIT, CHARS_PER_TOKEN
)
from app.db.vector_db import get_vector_db

logger = logging.getLogger(__name__)

TEMP_PDF_DIR = "/tmp/pdfs"
os.makedirs(TEMP_PDF_DIR, exist_ok=True)


async def summarize_qa(
    question: str, 
    answer: str, 
    llm_name: str, 
    context_window: int = DEFAULT_TOKEN_BUDGET
) -> str:
    """
    Summarize a QA pair using the LLM for concise memory storage or display.
    Uses dynamic hard limits based on context window percentages.
    """
    from langchain_core.messages import HumanMessage
    try:
        llm = get_llm(llm_name)
        prompt = (
            "Summarize the following Q&A pair accurately and concisely. "
            "Keep it under 3 sentences and preserve key facts.\n\n"
            f"Q: {question}\n"
            f"A: {answer}\n\n"
            "Summary:"
        )
        # Use simple invoke for summarization
        from app.agent.agent import invoke_with_retry
        response = await invoke_with_retry(llm.ainvoke, [HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error summarizing QA: {e}")
        # Fallback to truncated version based on hard limit percentage
        hard_limit_chars = int(context_window * RATIO_MEMORY_HARD_LIMIT * CHARS_PER_TOKEN)
        return f"Q: {question}\nA: {answer}"[:hard_limit_chars] + "..."


async def download_and_parse_pdf(file_hash: str, backend_url: str) -> Optional[List[str]]:
    """
    Download a PDF from the backend using file_hash and parse it into text chunks using unstructured.
    Returns a list of chunked strings, or None if download/parsing fails.
    """
    pdf_url = f"{backend_url}/{file_hash}.pdf"
    local_path = os.path.join(TEMP_PDF_DIR, f"{file_hash}.pdf")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(pdf_url, timeout=30.0)
            if resp.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(resp.content)
                
                # Run partitioning in a thread pool as it is CPU-bound
                elements = await asyncio.to_thread(partition_pdf, filename=local_path)
                
                from unstructured.chunking.title import chunk_by_title
                # Improved chunking: ensure sentences are not split and use consistent sizing
                # multipage_sections=True helps keep context across page breaks
                chunked_elements = chunk_by_title(
                    elements,
                    multipage_sections=True,
                    combine_text_under_n_chars=200,
                    max_characters=500,
                    new_after_n_chars=400,
                    overlap=0 # Neighbors provide the continuity, so we don't need overlapping text
                )
                chunks = [str(c) for c in chunked_elements]
                try:
                    os.remove(local_path)
                except Exception:
                    pass
                return chunks
            else:
                logger.error(f"Failed to download PDF from {pdf_url}: {resp.status_code}")
                return None
    except Exception as e:
        logger.error(f"Error downloading/parsing PDF: {e}")
        return None


async def get_chunks(file_hash: str) -> List[str]:
    """
    Download a PDF from the backend using file_hash and parse it into text chunks using unstructured.
    Returns a list of chunked strings.
    """
    backend_url = os.getenv("BACKEND_URL", "http://backend:8000")
    chunks = await download_and_parse_pdf(file_hash, backend_url)
    if chunks:
        return chunks
    
    logger.error(f"PDF download/parse failed for {file_hash}")
    return []


async def get_chat_chunks(
    question: str, 
    answer: str, 
    llm_name: Optional[str] = None,
    context_window: int = DEFAULT_TOKEN_BUDGET
) -> List[str]:
    """
    Format, optionally summarize, and chunk a QA pair into text snippets for memory retrieval.
    This utilizes LangChain splitters for consistent chunking and dynamic thresholding.
    """
    compact_text, _ = await build_compact_chat_memory_text(
        question=question,
        answer=answer,
        llm_name=llm_name,
        context_window=context_window,
    )
    return split_chat_memory_text(compact_text)


async def build_compact_chat_memory_text(
    question: str,
    answer: str,
    llm_name: Optional[str] = None,
    context_window: int = DEFAULT_TOKEN_BUDGET
) -> tuple[str, bool]:
    """
    Build compact QA text for semantic memory.
    Returns (text, was_summarized).
    """
    qa_text = f"Q: {question}\nA: {answer}"
    summarization_threshold_chars = int(context_window * RATIO_MEMORY_SUMMARIZATION_THRESHOLD * CHARS_PER_TOKEN)

    if len(qa_text) > summarization_threshold_chars and llm_name:
        logger.info(f"QA pair length ({len(qa_text)}) > threshold ({summarization_threshold_chars}), summarizing.")
        summary = await summarize_qa(question, answer, llm_name, context_window)
        return f"Q: {question}\nSummary: {summary}", True

    return qa_text, False


def split_chat_memory_text(compact_text: str) -> List[str]:
    """Chunk compact chat memory text for vector indexing."""
    doc = Document(page_content=compact_text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents([doc])
    return [c.page_content for c in chunks]


async def generate_embeddings(chunks: List[str], embedding_model_name: str) -> List[List[float]]:
    """
    Generate embeddings for each chunk using the specified embedding model.
    Note: Some LLM APIs/servers (like DMR) may have strict batch size limits.
    """
    embed_model = get_embedding_model(embedding_model_name)
    batch_size = 100  # LLM API/server strict batch size limits
    vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_vectors = await embed_model.aembed_documents(batch)
        vectors.extend(batch_vectors)
    return vectors


async def index_document_for_thread(
    thread_id: str,
    file_hash: str,
    embedding_model_name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Index a document into the vector database.
    The PDF is fetched from the backend and parsed using Unstructured.
    
    Args:
        thread_id: The thread ID to index into
        file_hash: Unique hash of the file
        embedding_model_name: The embedding model to use
        metadata: Additional metadata to store with chunks
    
    Returns:
        Status dict with indexing results
    """
    db_client = get_vector_db()
    metadata = metadata or {}
    
    try:
        if await db_client.has_file_indexed(thread_id, file_hash, embedding_model_name):
            try:
                from app.db.database import update_document_indexing_status
                shared_chunks = await db_client.get_file_chunk_count(file_hash, embedding_model_name)
                await update_document_indexing_status(
                    thread_id=thread_id,
                    file_hash=file_hash,
                    status="indexed",
                    chunk_count=shared_chunks,
                    total_chars=0,
                )
            except Exception:
                pass
            return {
                "status": "success",
                "thread_id": thread_id,
                "file_hash": file_hash,
                "chunks_count": await db_client.get_file_chunk_count(file_hash, embedding_model_name),
                "reused_existing_embeddings": True,
            }

        # 1. Get chunks from PDF
        chunks = await get_chunks(file_hash)
        if not chunks:
            logger.warning(f"No chunks extracted for thread {thread_id}, file {file_hash}")
            return {"status": "error", "message": "No text extracted from PDF"}
        
        logger.info(f"Extracted {len(chunks)} chunks for thread {thread_id}, file {file_hash}")
        
        # 2. Generate embeddings
        vectors = await generate_embeddings(chunks, embedding_model_name)
        
        # 3. Prepare metadata for each chunk
        chunk_metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                "file_hash": file_hash,
                "chunk_index": i,
            }
            chunk_metadatas.append(chunk_metadata)
        
        # 4. Index into thread collection
        indexed_count = await db_client.index_pdf_chunks(
            thread_id=thread_id,
            embedding_model_name=embedding_model_name,
            file_hash=file_hash,
            texts=chunks,
            embeddings=vectors,
            metadatas=chunk_metadatas
        )
        
        logger.info(f"Successfully indexed {indexed_count} chunks for thread {thread_id}")

        # Update thread stats snapshot
        try:
            from app.db.database import update_document_indexing_status
            await update_document_indexing_status(
                thread_id=thread_id,
                file_hash=file_hash,
                status="indexed",
                chunk_count=indexed_count,
                total_chars=sum(len(c) for c in chunks),
            )
        except Exception as stats_err:
            logger.warning(f"thread_stats update skipped after indexing: {stats_err}")

        return {
            "status": "success",
            "thread_id": thread_id,
            "file_hash": file_hash,
            "chunks_count": indexed_count
        }

    except Exception as e:
        logger.error(f"Error indexing document for thread {thread_id}: {e}", exc_info=True)
        try:
            from app.db.database import update_document_indexing_status
            await update_document_indexing_status(thread_id=thread_id, file_hash=file_hash, status="failed")
        except Exception:
            pass
        return {"status": "error", "message": str(e)}


async def index_chat_memory_for_thread(
    thread_id: str,
    message_id: str,
    question: str,
    answer: str,
    embedding_model_name: str,
    llm_name: Optional[str] = None,
    context_window: int = DEFAULT_TOKEN_BUDGET
) -> Dict[str, Any]:
    """
    Process, chunk, and index a chat message as semantic memory.
    Uses LangChain splitters and optional LLM summarization with dynamic budgets.
    """
    db_client = get_vector_db()
    
    try:
        # 1. Build compact memory text and chunk for indexing.
        compact_text, was_summarized = await build_compact_chat_memory_text(
            question=question,
            answer=answer,
            llm_name=llm_name,
            context_window=context_window,
        )
        chunks = split_chat_memory_text(compact_text)
        
        # 2. Generate embeddings
        vectors = await generate_embeddings(chunks, embedding_model_name)
        
        # 3. Store into vector database
        indexed_count = await db_client.index_chat_memory(
            thread_id=thread_id,
            message_id=message_id,
            question=question,
            answer=answer,
            texts=chunks,
            embeddings=vectors
        )
        
        return {
            "status": "success",
            "chunks_count": indexed_count,
            "memory_compact_text": compact_text,
            "memory_was_summarized": was_summarized,
        }
    except Exception as e:
        logger.error(f"Error indexing chat memory for thread {thread_id}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


def _extract_question_from_compact_text(compact_text: str) -> str:
    """Extract the leading `Q:` line from compact chat-memory text."""
    for line in compact_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Q:"):
            return line[2:].strip()
        break
    return ""


async def index_chat_memory_from_compact_for_thread(
    thread_id: str,
    message_id: str,
    compact_text: str,
    answer: str,
    embedding_model_name: str,
) -> Dict[str, Any]:
    """
    Re-index chat memory using the exact persisted compact text so summarized
    vs non-summarized memory stays identical to what was originally stored.
    """
    db_client = get_vector_db()
    try:
        chunks = split_chat_memory_text(compact_text)
        vectors = await generate_embeddings(chunks, embedding_model_name)
        indexed_count = await db_client.index_chat_memory(
            thread_id=thread_id,
            message_id=message_id,
            question=_extract_question_from_compact_text(compact_text),
            answer=answer,
            texts=chunks,
            embeddings=vectors,
        )
        return {"status": "success", "chunks_count": indexed_count}
    except Exception as e:
        logger.error(
            "Error re-indexing compact chat memory for thread %s message %s: %s",
            thread_id,
            message_id,
            e,
            exc_info=True,
        )
        return {"status": "error", "message": str(e)}


async def index_webpage_for_thread(
    thread_id: str,
    url: str,
    file_hash: str,
    embedding_model_name: str,
) -> Dict[str, Any]:
    """
    Index a webpage into vector storage.
    Snapshot-first behavior:
    1) Try `/static/webpages/{file_hash}.html` if present.
    2) Fallback to live URL fetch when snapshot is missing/unusable.

    Args:
        thread_id: The thread to index into.
        url: The full URL of the page to fetch.
        file_hash: MD5 hash of the URL (used as the stable identifier).
        embedding_model_name: Embedding model to use.

    Returns:
        Status dict with indexed chunk count, page title, and any error.
    """
    from bs4 import BeautifulSoup

    def _extract_text_and_title_from_html(html: str, fallback_title: str) -> tuple[str, str]:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else fallback_title
        content_tag = soup.find("main") or soup.find("article") or soup.body or soup
        raw_text = content_tag.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        return "\n".join(lines), title

    db_client = get_vector_db()
    try:
        if await db_client.has_file_indexed(thread_id, file_hash, embedding_model_name):
            try:
                from app.db.database import update_document_indexing_status
                shared_chunks = await db_client.get_file_chunk_count(file_hash, embedding_model_name)
                await update_document_indexing_status(
                    thread_id=thread_id,
                    file_hash=file_hash,
                    status="indexed",
                    chunk_count=shared_chunks,
                    total_chars=0,
                )
            except Exception:
                pass
            return {
                "status": "success",
                "thread_id": thread_id,
                "file_hash": file_hash,
                "url": url,
                "title": url,
                "chunks_count": await db_client.get_file_chunk_count(file_hash, embedding_model_name),
                "reused_existing_embeddings": True,
            }

        html = ""
        title = url
        snapshot_path = f"/static/webpages/{file_hash}.html"
        if os.path.exists(snapshot_path):
            try:
                with open(snapshot_path, "r", encoding="utf-8", errors="replace") as f:
                    html = f.read()
                text, title = _extract_text_and_title_from_html(html, url)
                logger.info("Using captured snapshot for web re-embed: %s", snapshot_path)
            except Exception as snapshot_err:
                logger.warning("Failed reading snapshot %s: %s", snapshot_path, snapshot_err)
                text = ""
        else:
            text = ""

        if not text:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; AskPDF bot)"})
                resp.raise_for_status()
                html = resp.text
            text, title = _extract_text_and_title_from_html(html, url)

        if not text:
            return {"status": "error", "message": "No text content found on page"}

        # Chunk the text
        doc = Document(page_content=text)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = [c.page_content for c in splitter.split_documents([doc])]

        if not chunks:
            return {"status": "error", "message": "No chunks produced from page text"}

        vectors = await generate_embeddings(chunks, embedding_model_name)

        indexed_count = await db_client.index_web_source_chunks(
            thread_id=thread_id,
            embedding_model_name=embedding_model_name,
            file_hash=file_hash,
            url=url,
            title=title,
            texts=chunks,
            embeddings=vectors,
        )

        logger.info(f"Indexed {indexed_count} web source chunks for thread {thread_id}, url {url}")

        # Update thread stats snapshot
        try:
            from app.db.database import update_document_indexing_status
            await update_document_indexing_status(
                thread_id=thread_id,
                file_hash=file_hash,
                status="indexed",
                chunk_count=indexed_count,
                total_chars=sum(len(c) for c in chunks),
            )
        except Exception as stats_err:
            logger.warning(f"thread_stats update skipped after web indexing: {stats_err}")

        return {
            "status": "success",
            "thread_id": thread_id,
            "file_hash": file_hash,
            "url": url,
            "title": title,
            "chunks_count": indexed_count,
        }

    except httpx.HTTPStatusError as e:
        msg = f"HTTP error fetching {url}: {e.response.status_code}"
        logger.error(msg)
        try:
            from app.db.database import update_document_indexing_status
            await update_document_indexing_status(thread_id=thread_id, file_hash=file_hash, status="failed")
        except Exception:
            pass
        return {"status": "error", "message": msg}
    except Exception as e:
        logger.error(f"Error indexing webpage for thread {thread_id}: {e}", exc_info=True)
        try:
            from app.db.database import update_document_indexing_status
            await update_document_indexing_status(thread_id=thread_id, file_hash=file_hash, status="failed")
        except Exception:
            pass
        return {"status": "error", "message": str(e)}


async def index_web_search_for_thread(
    thread_id: str,
    query: str,
    texts: List[str],
    urls: Optional[List[str]],
    titles: Optional[List[str]],
    embedding_model_name: str,
) -> Dict[str, Any]:
    """
    Embed and store web search result snippets into the thread's vector collection
    so they can be semantically retrieved in future queries.

    Args:
        thread_id: The thread to store snippets in.
        query: The search query that produced these results.
        texts: List of result snippet texts.
        urls: Corresponding source URLs (parallel to texts).
        titles: Corresponding page titles (parallel to texts).
        embedding_model_name: Embedding model to use.

    Returns:
        Status dict with indexed chunk count.
    """
    db_client = get_vector_db()
    try:
        vectors = await generate_embeddings(texts, embedding_model_name)
        indexed_count = await db_client.index_web_search_chunks(
            thread_id=thread_id,
            query=query,
            texts=texts,
            embeddings=vectors,
            urls=urls,
            titles=titles,
        )
        return {"status": "success", "chunks_count": indexed_count}
    except Exception as e:
        logger.error(f"Error indexing web search for thread {thread_id}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


_reembed_locks: Dict[str, asyncio.Lock] = {}


def _thread_reembed_lock(thread_id: str) -> asyncio.Lock:
    """Return the per-thread lock used to avoid concurrent re-embed runs."""
    if thread_id not in _reembed_locks:
        _reembed_locks[thread_id] = asyncio.Lock()
    return _reembed_locks[thread_id]


async def trigger_reembed_for_missing_sources(
    thread_id: str,
    embedding_model_name: str,
    file_hashes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Lazy backfill for sources and chat-memory vectors missing in vector DB.
    Called when a thread is opened.
    """
    from app.db.database import get_thread_files, get_thread_messages, MessageRole
    if not await embed_model_check(thread_id, embedding_model_name):
        return {"status": "skipped", "reason": "embed_model_not_ready"}

    lock = _thread_reembed_lock(thread_id)
    if lock.locked():
        return {"status": "skipped", "reason": "reembed_in_progress"}

    async with lock:
        files = await get_thread_files(thread_id)
        if file_hashes:
            wanted = set(file_hashes)
            files = [f for f in files if f.file_hash in wanted]

        db = get_vector_db()
        reindexed_files: List[Dict[str, str]] = []
        reindexed_chat_messages: List[str] = []

        for f in files:
            try:
                if not await embed_model_check(thread_id, embedding_model_name, during_run=True):
                    logger.warning(
                        "Stopping re-embed for thread %s: embed model '%s' became unavailable",
                        thread_id,
                        embedding_model_name,
                    )
                    break
                if await db.has_file_indexed(thread_id, f.file_hash, embedding_model_name):
                    continue
                if f.source_type == "web":
                    url = (f.file_path or f.file_name or "").strip()
                    if not url:
                        continue
                    result = await index_webpage_for_thread(
                        thread_id=thread_id,
                        url=url,
                        file_hash=f.file_hash,
                        embedding_model_name=embedding_model_name,
                    )
                    if result.get("status") == "success":
                        reindexed_files.append({"file_hash": f.file_hash, "source_type": "web"})
                else:
                    result = await index_document_for_thread(
                        thread_id=thread_id,
                        file_hash=f.file_hash,
                        embedding_model_name=embedding_model_name,
                    )
                    if result.get("status") == "success":
                        reindexed_files.append({"file_hash": f.file_hash, "source_type": "pdf"})
            except Exception as item_err:
                logger.warning("Skipping re-embed for file %s: %s", f.file_hash, item_err)

        try:
            messages = await get_thread_messages(thread_id, limit=10000)
            for msg in messages:
                if not await embed_model_check(thread_id, embedding_model_name, during_run=True):
                    logger.warning(
                        "Stopping chat-memory backfill for thread %s: embed model '%s' became unavailable",
                        thread_id,
                        embedding_model_name,
                    )
                    break
                if msg.role != MessageRole.ASSISTANT:
                    continue
                compact_text = (msg.context_compact or "").strip()
                if not compact_text:
                    continue
                if await db.has_chat_memory_indexed(thread_id, msg.id):
                    continue
                chat_result = await index_chat_memory_from_compact_for_thread(
                    thread_id=thread_id,
                    message_id=msg.id,
                    compact_text=compact_text,
                    answer=msg.content,
                    embedding_model_name=embedding_model_name,
                )
                if chat_result.get("status") == "success":
                    reindexed_chat_messages.append(msg.id)
        except Exception as chat_err:
            logger.warning("Skipping chat-memory backfill for thread %s: %s", thread_id, chat_err)

        return {
            "status": "completed",
            "file_count": len(reindexed_files),
            "chat_memory_count": len(reindexed_chat_messages),
            "files": reindexed_files,
            "chat_message_ids": reindexed_chat_messages,
        }


async def embed_model_check(
    thread_id: str, embedding_model_name: str, during_run: bool = False
) -> bool:
    """
    Verify embedding-model availability for re-index paths.
    Returns False on both hard not-ready and check failures.
    """
    from app.models.llm_server_client import check_embed_model_ready

    try:
        ready = await check_embed_model_ready(embedding_model_name, use_cache=False)
    except Exception as ready_err:
        phase = "during re-embed run" if during_run else "before re-embed trigger"
        logger.warning(
            "Skipping re-embed for thread %s: embed-model readiness check failed %s for '%s': %s",
            thread_id,
            phase,
            embedding_model_name,
            ready_err,
        )
        return False

    if not ready:
        logger.info(
            "Skipping re-embed for thread %s: embed model '%s' is not ready",
            thread_id,
            embedding_model_name,
        )
        return False

    return True
