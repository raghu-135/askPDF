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

from agent import (
    app as agent_app, 
    invoke_with_retry, 
    generate_optimized_search_query, 
    perform_web_search
)
from models import get_llm, get_embedding_model, get_system_prompt
from vectordb.qdrant import QdrantAdapter
from database import (
    create_message, get_recent_messages, get_thread,
    MessageRole
)

logger = logging.getLogger(__name__)

# Token budget configuration
DEFAULT_TOKEN_BUDGET = 4096  # Estimate for context window

# Context allocation ratios (must sum to 1.0)
RATIO_LLM_RESPONSE = 0.25      # Reserve 25% for answer
RATIO_PDF_CONTEXT = 0.40       # 40% for PDF chunks
RATIO_RECENT_HISTORY = 0.20    # 20% for recent conversation history
RATIO_SEMANTIC_MEMORY = 0.15   # 15% for recalled semantic memories

# Approximate tokens per character (rough estimate)
CHARS_PER_TOKEN = 4

# Max characters for embedding input (most models support 512-8192 tokens)
# Using conservative limit: ~400 tokens = ~1600 chars for safety
MAX_EMBEDDING_CHARS = 1600


def estimate_tokens(text: str) -> int:
    """Rough token estimate based on character count."""
    return len(text) // CHARS_PER_TOKEN


def truncate_for_embedding(text: str, max_chars: int = MAX_EMBEDDING_CHARS) -> str:
    """
    Truncate text to fit within embedding model's input limit.
    Tries to cut at sentence boundaries when possible.
    """
    if len(text) <= max_chars:
        return text
    
    # Try to cut at a sentence boundary
    truncated = text[:max_chars]
    
    # Find last sentence end within the truncated text
    for sep in ['. ', '? ', '! ', '\n']:
        last_sep = truncated.rfind(sep)
        if last_sep > max_chars // 2:  # Only cut if we keep at least half
            return truncated[:last_sep + 1].strip()
    
    # Fallback: cut at word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_chars // 2:
        return truncated[:last_space].strip() + "..."
    
    return truncated.strip() + "..."


def budget_context(
    pdf_chunks: List[Dict[str, Any]],
    recent_messages: List[Any],
    recalled_memories: List[Dict[str, Any]],
    question: str,
    max_tokens: int = DEFAULT_TOKEN_BUDGET
) -> Tuple[List[Dict[str, Any]], List[Any], List[Dict[str, Any]]]:
    """
    Budget the context to fit within token limits based on ratios.
    Priority is handled within each section's own budget.
    
    Returns filtered lists that fit within their respective budget.
    """
    # Calculate token caps based on ratios
    pdf_cap = int(max_tokens * RATIO_PDF_CONTEXT)
    history_cap = int(max_tokens * RATIO_RECENT_HISTORY)
    memory_cap = int(max_tokens * RATIO_SEMANTIC_MEMORY)
    
    selected_pdf_chunks = []
    selected_messages = []
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
    
    # 2. Fill recent messages (up to its allocated ratio)
    used_history_tokens = 0
    for msg in recent_messages:
        tokens = estimate_tokens(msg.content)
        if used_history_tokens + tokens <= history_cap:
            selected_messages.append(msg)
            used_history_tokens += tokens
        else:
            break
    
    # 3. Fill recalled memories (up to its allocated ratio)
    used_memory_tokens = 0
    for memory in recalled_memories:
        tokens = estimate_tokens(memory.get("text", ""))
        if used_memory_tokens + tokens <= memory_cap:
            selected_memories.append(memory)
            used_memory_tokens += tokens
        else:
            break
    
    return selected_pdf_chunks, selected_messages, selected_memories


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
        
        # Calculate dynamic search limits (Assuming avg ~150 tokens per item, fetch 2x for safety)
        pdf_limit = max(50, int((context_window * RATIO_PDF_CONTEXT) / 75))
        history_limit = max(4, int((context_window * RATIO_RECENT_HISTORY) / 50))
        memory_limit = max(30, int((context_window * RATIO_SEMANTIC_MEMORY) / 100))

        # 2. Search PDF chunks
        pdf_chunks = await db.search_pdf_chunks(
            thread_id=thread_id,
            query_vector=query_vector,
            limit=pdf_limit
        )
        
        # 3. Search chat memory (excluding current conversation)
        recent_msgs = await get_recent_messages(thread_id, limit=history_limit)
        recent_message_ids = [m.id for m in recent_msgs]
        
        recalled_memories = await db.search_chat_memory(
            thread_id=thread_id,
            query_vector=query_vector,
            limit=memory_limit,
            exclude_message_ids=recent_message_ids
        )
        
        # 4. Convert recent messages to LangChain format
        chat_history = []
        for msg in recent_msgs:
            if msg.role == MessageRole.USER:
                chat_history.append(HumanMessage(content=msg.content))
            else:
                chat_history.append(AIMessage(content=msg.content))
        
        # 5. Budget context
        selected_chunks, selected_history, selected_memories = budget_context(
            pdf_chunks=pdf_chunks,
            recent_messages=chat_history,
            recalled_memories=recalled_memories,
            question=question,
            max_tokens=context_window
        )
        
        # 6. Build context string
        pdf_context = "\n\n".join([chunk["text"] for chunk in selected_chunks])
        memory_context = ""
        if selected_memories:
            memory_context = "\n\n--- Relevant Past Conversations ---\n"
            memory_context += "\n".join([mem["text"] for mem in selected_memories])
        
        full_context = f"PDF Context:\n{pdf_context}"
        if memory_context:
            full_context += f"\n\n{memory_context}"
        
        # 7. Optional web search
        web_context = ""
        if use_web_search:
            # Generate optimized search query considering PDF context and history
            search_query = await generate_optimized_search_query(
                question=question,
                context=pdf_context,
                history=chat_history,
                llm_name=llm_model
            )
            
            web_results = await perform_web_search(search_query)
            web_context = f"\n\nWeb Search Results:\n{web_results}"
            full_context += web_context
        
        # 8. Generate response
        llm = get_llm(llm_model)
        
        system_instruction = get_system_prompt(
            context=full_context,
            use_history=bool(selected_memories),
            use_web=use_web_search
        )
        
        messages = [
            SystemMessage(content=system_instruction),
        ]
        
        # Add recent history
        messages.extend(selected_history)
        messages.append(HumanMessage(content=question))
        
        logger.info(f"Final message send to LLM: {messages}")
        logger.info(f"Context breakdown - PDF chunks: {len(selected_chunks)}, Recent messages: {len(selected_history)}, Recalled memories: {len(selected_memories)}, Web search: {'Yes' if use_web_search else 'No'}")
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
        # Embed the combined Q&A for future retrieval
        # Truncate to fit embedding model's input limit
        qa_text = f"Q: {question}\nA: {answer}"
        qa_text_truncated = truncate_for_embedding(qa_text)
        qa_embedding = await invoke_with_retry(embed_model_client.aembed_query, qa_text_truncated)
        
        await db.upsert_chat_memory(
            thread_id=thread_id,
            message_id=assistant_message.id,
            question=question,
            answer=answer,
            embedding=qa_embedding
        )
        
        # 11. Prepare response
        used_chat_ids = [mem["message_id"] for mem in selected_memories if mem.get("message_id")]
        pdf_sources = [
            {
                "text": chunk["text"][:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", ""),
                "file_hash": chunk.get("file_hash"),
                "score": chunk.get("score")
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
