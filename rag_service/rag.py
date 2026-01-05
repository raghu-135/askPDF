from typing import Dict, Any, List
import os
import httpx
from models import get_embedding_model
from vectordb.qdrant import QdrantAdapter
from unstructured.partition.pdf import partition_pdf

TEMP_PDF_DIR = "/tmp/pdfs"
os.makedirs(TEMP_PDF_DIR, exist_ok=True)

async def index_document(text: str, embedding_model_name: str, metadata: Dict[str, Any] = None):
    """
    Indexes a document. 
    If file_hash is present in metadata, it creates a unique collection and uses unstructured for parsing.
    Otherwise falls back to simple text indexing (legacy).
    """
    metadata = metadata or {}
    file_hash = metadata.get("file_hash")
    upload_id = metadata.get("upload_id")
    
    # 1. Determine Collection Name
    base_model_name = embedding_model_name.split(":")[0]
    safe_model_name = base_model_name.replace("-", "_").replace(".", "_").replace("/", "_")
    
    if file_hash:
        collection_name = f"rag_{safe_model_name}_{file_hash}"
    else:
        # Fallback to generic collection
        collection_name = f"rag_{safe_model_name}"

    db_client = QdrantAdapter()
    
    # Check if exists
    if await db_client.collection_exists(collection_name):
        print(f"Collection {collection_name} already exists. Skipping indexing.", flush=True)
        return {"status": "skipped", "reason": "exists", "collection": collection_name}

    # 2. Parsing & Chunking
    chunks = []
    
    if file_hash and upload_id:
        # Fetch file from backend
        backend_url = os.getenv("BACKEND_URL", "http://backend:8000")
        pdf_url = f"{backend_url}/{upload_id}.pdf"
        local_path = os.path.join(TEMP_PDF_DIR, f"{upload_id}.pdf")
        
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(pdf_url, timeout=30.0)
                if resp.status_code == 200:
                    with open(local_path, "wb") as f:
                        f.write(resp.content)
                        
                    # Use Unstructured
                    elements = partition_pdf(filename=local_path)
                    
                    # Simple chunking of elements
                    # Using unstructured's default chunking or converting to text first
                    # For now, let's aggregate text and use recursive splitter for consistency with embedding model
                    # (Or we could use unstructured's chunk_by_title)
                    from unstructured.chunking.title import chunk_by_title
                    chunked_elements = chunk_by_title(elements)
                    chunks = [str(c) for c in chunked_elements]
                    
                    # Cleanup manually if needed, or let tmp cleaner handle it
                    try:
                        os.remove(local_path)
                    except:
                        pass
                else:
                     print(f"Failed to download PDF from {pdf_url}: {resp.status_code}", flush=True)
                     # Fallback to provided text
                     from langchain_text_splitters import RecursiveCharacterTextSplitter
                     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                     chunks = splitter.split_text(text)
        except Exception as e:
            print(f"Error downloading/parsing PDF: {e}", flush=True)
            # Fallback
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_text(text)
    else:
        # Legacy path
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(text)
    
    if not chunks:
        return {"status": "error", "message": "No text extracted"}

    # 3. Embeddings
    embed_model = get_embedding_model(embedding_model_name)
    
    try:
        # Process embeddings one at a time due to DMR's strict batch size limits
        batch_size = 1
        vectors = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_vectors = await embed_model.aembed_documents(batch)
            vectors.extend(batch_vectors)
        
        # 4. Storage
        metadatas_list = [metadata for _ in chunks]
        
        await db_client.index_documents(collection_name, chunks, metadatas_list, vectors)
        
        return {"status": "success", "chunks_count": len(chunks), "collection": collection_name}
    except Exception as e:
        print(f"Error indexing: {e}", flush=True)
        return {"status": "error", "message": str(e)}
