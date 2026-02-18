"""
chat_service.py - Business logic for chat endpoints in RAG Service

This module provides:
- Legacy chat handling (handle_chat)
- Thread-based chat with semantic memory (handle_thread_chat)
- Token budgeting for context management
- Dual-search (PDF chunks + chat memory)
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag import index_chat_memory_for_thread, summarize_qa
from agent import (
    app as agent_app, 
    invoke_with_retry, 
    generate_optimized_search_query, 
    perform_web_search
)
from models import (
    get_llm, get_embedding_model, get_system_prompt,
    DEFAULT_TOKEN_BUDGET, RATIO_LLM_RESPONSE, RATIO_PDF_CONTEXT,
    RATIO_SEMANTIC_MEMORY, CHARS_PER_TOKEN,
    RATIO_MEMORY_SUMMARIZATION_THRESHOLD
)
from vectordb.qdrant import QdrantAdapter
from database import (
    create_message, get_thread,
    MessageRole
)

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Rough token estimate based on character count."""
    return len(text) // CHARS_PER_TOKEN


def budget_context(
    pdf_chunks: List[Dict[str, Any]],
    recalled_memories: List[Dict[str, Any]],
    question: str,
    max_tokens: int = DEFAULT_TOKEN_BUDGET
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Budget the context to fit within token limits based on ratios.
    Priority is handled within each section's own budget.
    
    Returns filtered lists that fit within their respective budget.
    """
    # Calculate token caps based on ratios
    pdf_cap = int(max_tokens * RATIO_PDF_CONTEXT)
    memory_cap = int(max_tokens * RATIO_SEMANTIC_MEMORY)
    
    selected_pdf_chunks = []
    selected_memories = []
    
    # 1. Fill PDF chunks (up to its allocated ratio)
    used_pdf_tokens = 0
    for chunk in pdf_chunks:
        tokens = estimate_tokens(chunk.get("text", ""))
        if used_pdf_tokens + tokens <= pdf_cap:
            selected_pdf_chunks.append(chunk)
            used_pdf_tokens += tokens
        else:
            break
    
    # 2. Fill recalled memories (up to its allocated ratio)
    used_memory_tokens = 0
    for memory in recalled_memories:
        tokens = estimate_tokens(memory.get("text", ""))
        if used_memory_tokens + tokens <= memory_cap:
            selected_memories.append(memory)
            used_memory_tokens += tokens
        else:
            break
    
    return selected_pdf_chunks, selected_memories


async def handle_chat(req) -> Dict[str, Any]:
    """
    Legacy chat handler for backward compatibility.
    Uses collection-based retrieval without semantic memory.
    """
    chat_history = []
    for msg in req.history:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))

    inputs = {
        "question": req.question,
        "chat_history": chat_history,
        "llm_model": req.llm_model,
        "embedding_model": req.embedding_model,
        "collection_name": req.collection_name,
        "use_web_search": req.use_web_search,
        "context": "",
        "web_context": "",
        "answer": "",
    }
    result = await agent_app.ainvoke(inputs)
    return {"answer": result["answer"], "context": result.get("context", "")}


async def handle_thread_chat(
    thread_id: str,
    req,  # ThreadChatRequest
    embed_model: str
) -> Dict[str, Any]:
    """
    Thread-based chat with dual-search and semantic memory.
    
    Flow:
    1. Embed the question
    2. Search PDF chunks in thread collection
    3. Search chat memory (past QA pairs)
    4. Get recent conversation history
    5. Budget context to fit token limits
    6. Generate response with LLM
    7. Store user and assistant messages
    8. Create semantic memory for the QA pair
    
    Returns:
        {
            "answer": str,
            "used_chat_ids": List[str],  # Message IDs that were recalled
            "pdf_sources": List[Dict],    # PDF chunks used
            "context": str                 # Combined context used
        }
    """
    db = QdrantAdapter()
    question = req.question
    llm_model = req.llm_model
    use_web_search = getattr(req, 'use_web_search', False)
    context_window = getattr(req, 'context_window', DEFAULT_TOKEN_BUDGET)
    
    try:
        # 1. Embed the question
        embed_model_client = get_embedding_model(embed_model)
        query_vector = await invoke_with_retry(embed_model_client.aembed_query, question)
        
        # Calculate dynamic search limits
        # We fetch fewer initial hits because we expand each one elastically
        pdf_limit = max(10, int((context_window * RATIO_PDF_CONTEXT) / 500))
        memory_limit = max(20, int((context_window * RATIO_SEMANTIC_MEMORY) / 100))

        # 2. Search PDF chunks
        raw_pdf_chunks = await db.search_pdf_chunks(
            thread_id=thread_id,
            query_vector=query_vector,
            limit=pdf_limit
        )

        # 2a. Elastic Neighbor Context Expansion
        # Radius expands based on context window: 4k -> 2, 32k -> 5, 128k -> 10
        expansion_radius = max(2, min(10, int(context_window / 8000) + 1))
        
        file_chunk_map = {}
        for hit in raw_pdf_chunks:
            file_hash = hit.get("file_hash")
            chunk_id = hit.get("chunk_id")
            if file_hash is not None and chunk_id is not None:
                if file_hash not in file_chunk_map:
                    file_chunk_map[file_hash] = set()
                
                # Dynamic range based on elastic radius
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
            
        # Re-sort expanded chunks by file_hash and chunk_id for logical flow
        expanded_pdf_chunks.sort(key=lambda x: (x.get("file_hash", ""), x.get("chunk_id", 0)))
        
        # 3. Search chat memory
        # We now rely entirely on semantic retrieval for past interactions
        recalled_memories = await db.search_chat_memory(
            thread_id=thread_id,
            query_vector=query_vector,
            limit=memory_limit
        )
        
        # 3a. Adaptive Memory Summarization
        # If memories are taking up too much of the context window, summarize them.
        summarization_threshold_chars = int(context_window * RATIO_MEMORY_SUMMARIZATION_THRESHOLD * CHARS_PER_TOKEN)
        
        for i, memory in enumerate(recalled_memories):
            if len(memory.get("text", "")) > summarization_threshold_chars:
                logger.debug(f"Memory {memory.get('message_id')} length ({len(memory.get('text', ''))}) > threshold ({summarization_threshold_chars}), summarizing.")
                summary = await summarize_qa(
                    question=memory.get("question", ""),
                    answer=memory.get("answer", ""),
                    llm_name=llm_model,
                    context_window=context_window
                )
                recalled_memories[i]["text"] = f"Q: {memory.get('question')}\nSummary: {summary}"

        # 4. Budget context
        selected_chunks, selected_memories = budget_context(
            pdf_chunks=expanded_pdf_chunks,
            recalled_memories=recalled_memories,
            question=question,
            max_tokens=context_window
        )
        
        # 5. Build context string
        pdf_context = "\n\n".join([chunk["text"] for chunk in selected_chunks])
        memory_context = ""
        if selected_memories:
            memory_context = "\n\n--- Relevant Past Conversations ---\n"
            memory_context += "\n".join([mem["text"] for mem in selected_memories])
        
        full_context = f"PDF Context:\n{pdf_context}"
        if memory_context:
            full_context += f"\n\n{memory_context}"
        
        # 6. Optional web search
        web_context = ""
        if use_web_search:
            # Generate optimized search query considering PDF context and history
            # we use an empty list for history as we've removed recent_messages
            search_query = await generate_optimized_search_query(
                question=question,
                context=pdf_context,
                history=[],
                llm_name=llm_model
            )
            
            web_results = await perform_web_search(search_query)
            web_context = f"\n\nWeb Search Results:\n{web_results}"
            full_context += web_context
        
        # 7. Generate response
        llm = get_llm(llm_model)
        
        system_instruction = get_system_prompt(
            context=full_context,
            use_history=bool(selected_memories),
            use_web=use_web_search
        )
        
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=question)
        ]

        logger.debug(f"Final message send to LLM: {messages}")
        logger.info(f"Context breakdown - PDF chunks: {len(selected_chunks)}, Recalled memories: {len(selected_memories)}, Web search: {'Yes' if use_web_search else 'No'}")
        response = await invoke_with_retry(llm.ainvoke, messages)
        answer = response.content
        
        # 9. Store messages in database
        user_message = await create_message(
            thread_id=thread_id,
            role=MessageRole.USER,
            content=question
        )
        
        assistant_message = await create_message(
            thread_id=thread_id,
            role=MessageRole.ASSISTANT,
            content=answer
        )
        
        # 10. Create semantic memory for this QA pair
        # Use LangChain-backed management with optional summarization
        await index_chat_memory_for_thread(
            thread_id=thread_id,
            message_id=assistant_message.id,
            question=question,
            answer=answer,
            embedding_model_name=embed_model,
            llm_name=llm_model,
            context_window=context_window
        )
        
        # 11. Prepare response
        used_chat_ids = [mem["message_id"] for mem in selected_memories if mem.get("message_id")]
        pdf_sources = [
            {
                "text": chunk["text"][:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", ""),
                "file_hash": chunk.get("file_hash"),
                "score": chunk.get("score", 0.0)  # Default to 0.0 for neighbor context
            }
            for chunk in selected_chunks
        ]
        
        return {
            "answer": answer,
            "user_message_id": user_message.id,
            "assistant_message_id": assistant_message.id,
            "used_chat_ids": used_chat_ids,
            "pdf_sources": pdf_sources,
            "context": full_context[:1000] + "..." if len(full_context) > 1000 else full_context
        }
        
    except Exception as e:
        logger.error("Error in handle_thread_chat", exc_info=True)
        raise e
