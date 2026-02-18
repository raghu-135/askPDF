"""
rag.py - Document indexing for RAG Service

This module handles:
- Text chunking and embedding generation
- Legacy collection-based indexing
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

from models import (
    get_embedding_model, get_llm, 
    DEFAULT_TOKEN_BUDGET, RATIO_MEMORY_SUMMARIZATION_THRESHOLD, 
    RATIO_MEMORY_HARD_LIMIT, CHARS_PER_TOKEN
)
from vectordb.qdrant import QdrantAdapter

logger = logging.getLogger(__name__)

TEMP_PDF_DIR = "/tmp/pdfs"
os.makedirs(TEMP_PDF_DIR, exist_ok=True)


def get_collection_name(embedding_model_name: str, file_hash: Optional[str] = None) -> str:
    """
    Generate a safe collection name for legacy vector database indexing.
    """
    base_model_name = embedding_model_name.split(":")[0]
    safe_model_name = base_model_name.replace("-", "_").replace(".", "_").replace("/", "_")
    if file_hash:
        return f"rag_{safe_model_name}_{file_hash}"
    return f"rag_{safe_model_name}"


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """
    Split text into chunks using RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


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
        from agent import invoke_with_retry
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
    qa_text = f"Q: {question}\nA: {answer}"
    
    # Calculate summarization threshold based on percentage of target context window
    summarization_threshold_chars = int(context_window * RATIO_MEMORY_SUMMARIZATION_THRESHOLD * CHARS_PER_TOKEN)

    # If the QA pair is taking too much budget, we summarize it first
    if len(qa_text) > summarization_threshold_chars and llm_name:
        logger.info(f"QA pair length ({len(qa_text)}) > threshold ({summarization_threshold_chars}), summarizing.")
        summary = await summarize_qa(question, answer, llm_name, context_window)
        qa_text = f"Q: {question}\nSummary: {summary}"
    
    # Create LangChain Document objects for more complex metadata/future usage
    doc = Document(page_content=qa_text)
    
    # Use LangChain's splitter directly on the document
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents([doc])
    
    return [c.page_content for c in chunks]


async def generate_embeddings(chunks: List[str], embedding_model_name: str) -> List[List[float]]:
    """
    Generate embeddings for each chunk using the specified embedding model.
    Note: Some LLM APIs/servers (like DMR) may have strict batch size limits.
    """
    embed_model = get_embedding_model(embedding_model_name)
    batch_size = 1  # LLM API/server strict batch size limits
    vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_vectors = await embed_model.aembed_documents(batch)
        vectors.extend(batch_vectors)
    return vectors


async def index_chunks_to_db(
    collection_name: str, 
    chunks: List[str], 
    metadatas_list: List[Dict[str, Any]], 
    vectors: List[List[float]], 
    db_client: QdrantAdapter
) -> None:
    """
    Index the chunks and their embeddings into the vector database.
    """
    await db_client.index_documents(collection_name, chunks, metadatas_list, vectors)


async def index_document(embedding_model_name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Legacy: Indexes a document into the vector database.
    Creates a collection based on embedding model and file hash.
    """
    metadata = metadata or {}
    file_hash = metadata.get("file_hash")
    
    if not file_hash:
        return {"status": "error", "message": "file_hash is required for indexing"}

    # 1. Determine Collection Name
    collection_name = get_collection_name(embedding_model_name, file_hash)
    db_client = QdrantAdapter()

    # 2. Check if collection exists
    if await db_client.collection_exists(collection_name):
        logger.info(f"Collection {collection_name} already exists. Skipping indexing.")
        return {"status": "skipped", "reason": "exists", "collection": collection_name}

    # 3. Parsing & Chunking
    chunks = await get_chunks(file_hash)
    if not chunks:
        return {"status": "error", "message": "No text extracted from PDF"}

    # 4. Embeddings
    try:
        vectors = await generate_embeddings(chunks, embedding_model_name)
        # 5. Storage
        metadatas_list = [metadata for _ in chunks]
        await index_chunks_to_db(collection_name, chunks, metadatas_list, vectors, db_client)
        return {"status": "success", "chunks_count": len(chunks), "collection": collection_name}
    except Exception as e:
        logger.error(f"Error indexing: {e}")
        return {"status": "error", "message": str(e)}


async def index_document_for_thread(
    thread_id: str,
    file_hash: str,
    embedding_model_name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Index a document into a thread's Qdrant collection.
    The PDF is fetched from the backend and parsed using Unstructured.
    
    Args:
        thread_id: The thread ID to index into
        file_hash: Unique hash of the file
        embedding_model_name: The embedding model to use
        metadata: Additional metadata to store with chunks
    
    Returns:
        Status dict with indexing results
    """
    db_client = QdrantAdapter()
    metadata = metadata or {}
    
    try:
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
            file_hash=file_hash,
            texts=chunks,
            embeddings=vectors,
            metadatas=chunk_metadatas
        )
        
        logger.info(f"Successfully indexed {indexed_count} chunks for thread {thread_id}")
        
        return {
            "status": "success",
            "thread_id": thread_id,
            "file_hash": file_hash,
            "chunks_count": indexed_count
        }
        
    except Exception as e:
        logger.error(f"Error indexing document for thread {thread_id}: {e}", exc_info=True)
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
    db_client = QdrantAdapter()
    
    try:
        # 1. Get chunks for the chat message (with optional summarization)
        chunks = await get_chat_chunks(question, answer, llm_name, context_window)
        
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
            "chunks_count": indexed_count
        }
    except Exception as e:
        logger.error(f"Error indexing chat memory for thread {thread_id}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


async def check_file_indexed_in_thread(thread_id: str, file_hash: str) -> bool:
    """
    Check if a file has been indexed in a thread's collection.
    """
    db_client = QdrantAdapter()
    return await db_client.has_file_indexed(thread_id, file_hash)
