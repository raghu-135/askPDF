from typing import Dict, Any
from models import get_embedding_model
from vectordb.qdrant import QdrantAdapter

async def index_text(text: str, embedding_model_name: str, metadata: Dict[str, Any] = None):
    """
    Chunks text, generates embeddings with specific model, and indexes into VectorDB.
    Collection name is derived from embedding_model_name to ensure compatibility.
    """
    # 1. Chunking - Use smaller chunks for DMR compatibility
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    
    # 2. Embeddings - Process in batches to avoid DMR batch size limits
    embed_model = get_embedding_model(embedding_model_name)
    
    # Process embeddings one at a time due to DMR's strict batch size limits
    batch_size = 1
    vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_vectors = await embed_model.aembed_documents(batch)
        vectors.extend(batch_vectors)
    
    # 3. Storage
    # Collection name safe handling
    # Strip tag from model name if present
    base_model_name = embedding_model_name.split(":")[0]
    safe_name = base_model_name.replace("-", "_").replace(".", "_").replace("/", "_")
    collection_name = f"rag_{safe_name}"
    
    # Metadata
    metadatas = [metadata or {} for _ in chunks]
    
    db_client = QdrantAdapter() # In future, factory for DBs
    await db_client.index_documents(collection_name, chunks, metadatas, vectors)
    
    return {"status": "success", "chunks_count": len(chunks), "collection": collection_name}
