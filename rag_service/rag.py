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
import time
from typing import Dict, Any, List, Optional

import httpx
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import (
    get_env_required,
    get_env_int_required,
    get_embedding_model, 
    DEFAULT_EMBEDDING_MODEL,
    GPU_EMBEDDING_BATCH_SIZE,
    LOCAL_EMBEDDING_MODELS,
    get_llm, 
    DEFAULT_TOKEN_BUDGET, RATIO_MEMORY_SUMMARIZATION_THRESHOLD, 
    RATIO_MEMORY_HARD_LIMIT, CHARS_PER_TOKEN
)
from vectordb.qdrant import get_qdrant

logger = logging.getLogger(__name__)

TEMP_PDF_DIR = "/tmp/pdfs"
os.makedirs(TEMP_PDF_DIR, exist_ok=True)


async def summarize_text(
    text: str,
    llm_name: str,
    max_sentences: int = 3,
    context_window: int = DEFAULT_TOKEN_BUDGET,
    instruction: str = "Summarize the following text accurately and concisely."
) -> str:
    """
    Generic text summarization using the LLM.
    """
    from langchain_core.messages import HumanMessage
    try:
        llm = get_llm(llm_name)
        prompt = (
            f"{instruction} "
            f"Keep it under {max_sentences} sentences and preserve key facts.\n\n"
            f"Text:\n{text}\n\n"
            "Summary:"
        )
        # Use simple invoke for summarization
        from agent import invoke_with_retry
        response = await invoke_with_retry(llm.ainvoke, [HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        # Fallback to truncated version based on hard limit percentage
        hard_limit_chars = int(context_window * RATIO_MEMORY_HARD_LIMIT * CHARS_PER_TOKEN)
        return text[:hard_limit_chars] + "..."


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
    qa_text = f"Q: {question}\nA: {answer}"
    return await summarize_text(
        text=qa_text,
        llm_name=llm_name,
        max_sentences=3,
        context_window=context_window,
        instruction="Summarize the following Q&A pair accurately and concisely."
    )


async def download_and_parse_pdf(file_hash: str, backend_url: str) -> Optional[List[Dict[str, Any]]]:
    """
    Download a PDF from the backend using file_hash and parse it into text chunks using unstructured.
    Returns a list of chunked data (text and metadata), or None if download/parsing fails.
    
    Optimizations:
    - Table structure inference (HTML -> Markdown)
    - Section-aware header injection (Contextual RAG)
    - Rich element metadata (Page, Category, Coordinates)
    """
    pdf_url = f"{backend_url}/{file_hash}.pdf"
    local_path = os.path.join(TEMP_PDF_DIR, f"{file_hash}.pdf")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(pdf_url, timeout=45.0)
            if resp.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(resp.content)
                
                # Run partitioning in a thread pool with high-fidelity settings
                elements = await asyncio.to_thread(
                    partition_pdf, 
                    filename=local_path,
                    infer_table_structure=True, # Extract structural table data
                    strategy="fast" # "fast" is good for throughput, "hi_res" is better for layout
                )
                
                # Pre-process elements to assist with header tracking and table conversion
                processed_elements = []
                current_section = None
                
                for el in elements:
                    category = getattr(el, 'category', 'Text')
                    if category == "Title":
                        current_section = el.text.strip()
                    
                    # Store tracking info on the element for the chunker
                    el.metadata.section = current_section
                    
                    # Convert tables to markdown if HTML is available
                    if category == "Table" and hasattr(el.metadata, 'text_as_html'):
                        try:
                            # Lightweight HTML to Markdown table conversion
                            import pandas as pd
                            import io
                            dfs = pd.read_html(io.StringIO(el.metadata.text_as_html))
                            if dfs:
                                el.text = dfs[0].to_markdown(index=False)
                        except Exception:
                            logger.warning("Table markdown conversion failed, falling back to raw text.")
                    
                    processed_elements.append(el)

                from unstructured.chunking.title import chunk_by_title
                chunked_elements = chunk_by_title(
                    processed_elements,
                    multipage_sections=True,
                    combine_text_under_n_chars=200,
                    max_characters=800, # Slightly larger chunks for better context
                    new_after_n_chars=600,
                    overlap=0 
                )
                
                chunks = []
                for c in chunked_elements:
                    page_number = getattr(c.metadata, 'page_number', None)
                    section = getattr(c.metadata, 'section', None)
                    category = getattr(c, 'category', 'CompositeElement')
                    
                    # Header Injection: Prepend section title if not already the first thing in the chunk
                    raw_text = str(c).strip()
                    if section and not raw_text.startswith(section):
                        injected_text = f"[Section: {section}]\n{raw_text}"
                    else:
                        injected_text = raw_text

                    chunks.append({
                        "text": injected_text,
                        "page_number": page_number,
                        "metadata": {
                            "section": section,
                            "category": category,
                            "is_table": category == "Table"
                        }
                    })

                try:
                    os.remove(local_path)
                except Exception:
                    pass
                return chunks
            else:
                logger.error(f"Failed to download PDF from {pdf_url}: {resp.status_code}")
                return None
    except Exception as e:
        logger.error(f"Error downloading/parsing PDF: {e}", exc_info=True)
        return None


async def get_chunks(file_hash: str) -> List[Dict[str, Any]]:
    """
    Retrieve text chunks based on backend-provided sentences if available,
    otherwise fallback to unstructured-based parsing. 
    This ensures that vector-based highlights exactly match the backend's sentence IDs.
    """
    backend_url = get_env_required("BACKEND_URL")
    
    # Try fetching sentences from backend cache first
    try:
        cache_url = f"{backend_url}/cache/{file_hash}.json"
        async with httpx.AsyncClient() as client:
            resp = await client.get(cache_url, timeout=10.0)
            if resp.status_code == 200:
                sentences = resp.json()
                logger.info(f"Using {len(sentences)} backend sentences to build chunks for {file_hash}")
                
                chunks: List[Dict[str, Any]] = []
                current_chunk_text = ""
                current_chunk_ids = []
                current_page = None
                
                # Logical chunking: group sentences while respecting page boundaries and size
                for s in sentences:
                    s_text = (s.get("text") or "").strip()
                    if not s_text: continue
                    s_id = s.get("id")
                    
                    # Estimate page number from bboxes (bboxes are from PyMuPDF, 1-indexed)
                    s_page = None
                    if s.get("bboxes") and len(s["bboxes"]) > 0:
                        s_page = s["bboxes"][0].get("page")
                    
                    # Start new chunk if page changes OR size exceeds target (approx 700 chars)
                    should_split = (current_page is not None and s_page != current_page) or \
                                 (len(current_chunk_text) + len(s_text) > 700)
                    
                    if should_split and current_chunk_text:
                        chunks.append({
                            "text": current_chunk_text.strip(),
                            "page_number": current_page,
                            "sentence_ids": list(current_chunk_ids)
                        })
                        current_chunk_text = ""
                        current_chunk_ids = []
                    
                    current_chunk_text += s_text + " "
                    if s_id is not None:
                        current_chunk_ids.append(s_id)
                    current_page = s_page
                
                if current_chunk_text:
                    chunks.append({
                        "text": current_chunk_text.strip(),
                        "page_number": current_page,
                        "sentence_ids": list(current_chunk_ids)
                    })
                return chunks
    except Exception as exc:
        logger.warning(f"Failed to fetch chunks from backend cache for {file_hash}: {exc}")

    # Fallback to pure unstructured parsing
    logger.info(f"Falling back to unstructured partitioner for {file_hash}")
    chunks = await download_and_parse_pdf(file_hash, backend_url)
    if chunks:
        return chunks
    
    logger.error(f"Both backend cache and unstructured failed for {file_hash}")
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


async def generate_hybrid_embeddings(chunks: List[str], embedding_model_name: str) -> tuple[List[List[float]], List[Dict[int, float]]]:
    """
    Generate both dense and sparse embeddings for each chunk using FastEmbed with batching.
    """
    logger.info(f"Generating hybrid embeddings for {len(chunks)} chunks using {embedding_model_name}...")
    embed_model = await get_embedding_model(embedding_model_name)
    
    if embedding_model_name in LOCAL_EMBEDDING_MODELS:
        batch_size = 50
    else:
        batch_size = GPU_EMBEDDING_BATCH_SIZE

    dense_vectors = []
    sparse_vectors = []
    
    total_chunks = len(chunks)
    total_batches = (total_chunks + batch_size - 1) // batch_size
    
    start_time_all = time.time()
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        current_batch = i // batch_size + 1
        logger.info(f"Processing batch {current_batch}/{total_batches} ({len(batch)} chunks)...")
        
        # Generate dense embeddings
        batch_dense = await embed_model.aembed_documents(batch)
        dense_vectors.extend(batch_dense)
        
        # Generate sparse embeddings
        batch_sparse = await embed_model.aembed_sparse_documents(batch)
        sparse_vectors.extend(batch_sparse)
        
    logger.info(f"Finished generating hybrid embeddings for all batches in {time.time() - start_time_all:.2f}s")
    return dense_vectors, sparse_vectors


async def index_document_for_thread(
    thread_id: str,
    file_hash: str,
    embedding_model_name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Index a document into a thread's Qdrant collection with Hybrid (Dense + BM25) support.
    """
    db_client = get_qdrant()
    metadata = metadata or {}
    
    try:
        # 1. Get chunks from PDF
        chunk_data = await get_chunks(file_hash)
        if not chunk_data:
            logger.warning(f"No chunks extracted for thread {thread_id}, file {file_hash}")
            return {"status": "error", "message": "No text extracted from PDF"}
        
        logger.info(f"Extracted {len(chunk_data)} chunks for thread {thread_id}, file {file_hash}")
        
        texts = [c["text"] for c in chunk_data]
        
        # 2. Generate Hybrid embeddings
        vectors, sparse_vectors = await generate_hybrid_embeddings(texts, embedding_model_name)
        
        # 3. Prepare metadata for each chunk
        chunk_metadatas = []
        for i, c in enumerate(chunk_data):
            chunk_metadata = {
                **metadata,
                "file_hash": file_hash,
                "chunk_index": i,
                "page_number": c.get("page_number"),
                "sentence_ids": c.get("sentence_ids"),
            }
            chunk_metadatas.append(chunk_metadata)
        
        # 4. Index into thread collection
        indexed_count = await db_client.index_pdf_chunks(
            thread_id=thread_id,
            file_hash=file_hash,
            texts=texts,
            embeddings=vectors,
            sparse_embeddings=sparse_vectors,
            metadatas=chunk_metadatas
        )
        
        logger.info(f"Successfully indexed {indexed_count} chunks for thread {thread_id}")

        # Update thread stats snapshot
        try:
            from database import update_document_indexing_status
            await update_document_indexing_status(
                thread_id=thread_id,
                file_hash=file_hash,
                status="indexed",
                chunk_count=indexed_count,
                total_chars=sum(len(t) for t in texts),
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
            from database import update_document_indexing_status
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
    Process, chunk, and index a chat message as hybrid semantic memory.
    """
    db_client = get_qdrant()
    
    try:
        # 1. Build compact memory text and chunk for indexing.
        compact_text, was_summarized = await build_compact_chat_memory_text(
            question=question,
            answer=answer,
            llm_name=llm_name,
            context_window=context_window,
        )
        chunks = split_chat_memory_text(compact_text)
        
        # 2. Generate Hybrid embeddings
        vectors, sparse_vectors = await generate_hybrid_embeddings(chunks, embedding_model_name)
        
        # 3. Store into vector database
        indexed_count = await db_client.index_chat_memory(
            thread_id=thread_id,
            message_id=message_id,
            question=question,
            answer=answer,
            texts=chunks,
            embeddings=vectors,
            sparse_embeddings=sparse_vectors
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


async def index_webpage_for_thread(
    thread_id: str,
    url: str,
    file_hash: str,
    embedding_model_name: str,
) -> Dict[str, Any]:
    """
    Index webpage content chunks with Hybrid support.
    """
    from bs4 import BeautifulSoup

    db_client = get_qdrant()
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; AskPDF bot)"})
            resp.raise_for_status()
            html = resp.text

        soup = BeautifulSoup(html, "lxml")

        # Remove non-content tags
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else url

        # Prefer main/article content, fall back to body
        content_tag = soup.find("main") or soup.find("article") or soup.body or soup
        raw_text = content_tag.get_text(separator="\n", strip=True)

        # Collapse excessive blank lines
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        text = "\n".join(lines)

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

        # Hybrid embeddings
        vectors, sparse_vectors = await generate_hybrid_embeddings(chunks, embedding_model_name)

        indexed_count = await db_client.index_web_source_chunks(
            thread_id=thread_id,
            file_hash=file_hash,
            url=url,
            title=title,
            texts=chunks,
            embeddings=vectors,
            sparse_embeddings=sparse_vectors
        )

        logger.info(f"Indexed {indexed_count} web source chunks for thread {thread_id}, url {url}")

        # Update thread stats snapshot
        try:
            from database import update_document_indexing_status
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
            from database import update_document_indexing_status
            await update_document_indexing_status(thread_id=thread_id, file_hash=file_hash, status="failed")
        except Exception:
            pass
        return {"status": "error", "message": msg}
    except Exception as e:
        logger.error(f"Error indexing webpage for thread {thread_id}: {e}", exc_info=True)
        try:
            from database import update_document_indexing_status
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
    Embed and store web search result snippets with Hybrid support.
    """
    db_client = get_qdrant()
    try:
        vectors, sparse_vectors = await generate_hybrid_embeddings(texts, embedding_model_name)
        indexed_count = await db_client.index_web_search_chunks(
            thread_id=thread_id,
            query=query,
            texts=texts,
            embeddings=vectors,
            sparse_embeddings=sparse_vectors,
            urls=urls,
            titles=titles,
        )
        return {"status": "success", "chunks_count": indexed_count}
    except Exception as e:
        logger.error(f"Error indexing web search for thread {thread_id}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
