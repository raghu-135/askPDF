import sys
import os
sys.path.append(os.path.abspath('rag_service'))

import asyncio
from vectordb.qdrant import get_qdrant

async def test():
    db = get_qdrant()
    # create test collection
    tid = "test_thread"
    collection_name = await db.create_thread_collection(tid)
    print(f"Collection: {collection_name}")
    
    # index a webpage
    fhash = "test_file_hash_123"
    await db.index_web_source_chunks(
        thread_id=tid,
        file_hash=fhash,
        url="http://example.com",
        title="Test page",
        texts=["hello world"],
        embeddings=[[0.1]*768]
    )
    
    # check indexed
    has = await db.has_file_indexed(tid, fhash)
    print(f"Has file indexed: {has}")
    
    # delete
    await db.delete_source_chunks_by_file_hash(tid, fhash)
    
    # check after delete
    has_after = await db.has_file_indexed(tid, fhash)
    print(f"Has file indexed after delete: {has_after}")
    
    # cleanup
    await db.delete_thread_collection(tid)

if __name__ == "__main__":
    os.environ['QDRANT_HOST'] = "localhost" # Assuming it might be running locally, wait, no.
    # We should just review the code.
