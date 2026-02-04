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
from typing import Dict, Any, List, Optional

import httpx
from unstructured.partition.pdf import partition_pdf

from models import get_embedding_model
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


def split_text(text: str) -> List[str]:
    """
    Split text into chunks using RecursiveCharacterTextSplitter.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)


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
                elements = partition_pdf(filename=local_path)
                from unstructured.chunking.title import chunk_by_title
                chunked_elements = chunk_by_title(elements)
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


async def get_chunks(text: str, file_hash: Optional[str]) -> List[str]:
    """
    Get text chunks from either a PDF (if file_hash is present) or from plain text.
    """
    if file_hash:
        backend_url = os.getenv("BACKEND_URL", "http://backend:8000")
        chunks = await download_and_parse_pdf(file_hash, backend_url)
        if chunks:
            return chunks
        # fallback to text splitting if PDF fails
        return split_text(text)
    else:
        return split_text(text)


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


async def index_document(text: str, embedding_model_name: str, metadata: Dict[str, Any] = None):
    """
    Legacy: Indexes a document into the vector database.
    Creates a collection based on embedding model and file hash.
    """
    metadata = metadata or {}
    file_hash = metadata.get("file_hash")

    # 1. Determine Collection Name
    collection_name = get_collection_name(embedding_model_name, file_hash)
    db_client = QdrantAdapter()

    # 2. Check if collection exists
    if await db_client.collection_exists(collection_name):
        logger.info(f"Collection {collection_name} already exists. Skipping indexing.")
        return {"status": "skipped", "reason": "exists", "collection": collection_name}

    # 3. Parsing & Chunking
    chunks = await get_chunks(text, file_hash)
    if not chunks:
        return {"status": "error", "message": "No text extracted"}

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
    text: str,
    embedding_model_name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Index a document into a thread's Qdrant collection.
    This is used for per-thread indexing with semantic memory support.
    
    Args:
        thread_id: The thread ID to index into
        file_hash: Unique hash of the file
        text: Document text (fallback if PDF parsing fails)
        embedding_model_name: The embedding model to use
        metadata: Additional metadata to store with chunks
    
    Returns:
        Status dict with indexing results
    """
    db_client = QdrantAdapter()
    metadata = metadata or {}
    
    try:
        # 1. Get chunks
        chunks = await get_chunks(text, file_hash)
        if not chunks:
            logger.warning(f"No chunks extracted for thread {thread_id}, file {file_hash}")
            return {"status": "error", "message": "No text extracted"}
        
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


async def check_file_indexed_in_thread(thread_id: str, file_hash: str) -> bool:
    """
    Check if a file has been indexed in a thread's collection.
    """
    db_client = QdrantAdapter()
    return await db_client.has_file_indexed(thread_id, file_hash)
