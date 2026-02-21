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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

from rag import index_chat_memory_for_thread, summarize_qa
from agent import (
    app as agent_app
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
    MessageRole, get_recent_messages
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
    """
    messages = []
    for msg in req.history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    config = {
        "configurable": {
            "thread_id": req.collection_name,
            "embedding_model": req.embedding_model,
            "llm_model": req.llm_model
        }
    }
    
    initial_state = {
        "messages": messages + [HumanMessage(content=req.question)],
        "llm_model": req.llm_model,
        "embedding_model": req.embedding_model,
        "thread_id": req.collection_name
    }
    
    result = await agent_app.ainvoke(initial_state, config=config)
    final_message = result["messages"][-1]
    
    return {"answer": final_message.content, "context": "Managed by agent"}


async def handle_thread_chat(
    thread_id: str,
    req,  # ThreadChatRequest
    embed_model: str
) -> Dict[str, Any]:
    """
    Thread-based chat using LangGraph agent for intelligent tool-calling and context management.
    """
    question = req.question
    llm_model = req.llm_model
    context_window = getattr(req, 'context_window', DEFAULT_TOKEN_BUDGET)
    
    try:
        # Configuration for tools and recursion
        config = {
            "configurable": {
                "thread_id": thread_id,
                "embedding_model": embed_model,
                "llm_model": llm_model
            }
        }
        
        # Load the last 15 messages to ground the agent in immediate context
        recent_history = await get_recent_messages(thread_id, limit=15)
        history_as_messages: List[BaseMessage] = []
        for msg in recent_history:
            if msg.role == MessageRole.USER:
                history_as_messages.append(HumanMessage(content=msg.content))
            else:
                history_as_messages.append(AIMessage(content=msg.content))
        
        # We start by grounding with the recent history
        initial_state = {
            "messages": history_as_messages + [HumanMessage(content=question)],
            "llm_model": llm_model,
            "embedding_model": embed_model,
            "thread_id": thread_id
        }
        
        # Invoke the agent graph
        logger.info(f"Invoking agent for thread_id={thread_id} with question: {question}")
        result = await agent_app.ainvoke(initial_state, config=config)
        
        # Get the final answer from last message in flow
        final_message = result["messages"][-1]
        answer = final_message.content
        
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
        await index_chat_memory_for_thread(
            thread_id=thread_id,
            message_id=assistant_message.id,
            question=question,
            answer=answer,
            embedding_model_name=embed_model,
            llm_name=llm_model,
            context_window=context_window
        )
        
        # Prepare returning metadata
        # We can extract used tool outputs from result["messages"] if we need specifically identified sources
        used_chat_ids = []
        pdf_sources = []
        
        # Extract metadata from tool outputs for debugging/display
        for msg in result["messages"]:
            if hasattr(msg, "tool_output") or (isinstance(msg, (AIMessage)) and hasattr(msg, "tool_calls")):
                continue # simplified
            # If we had custom attributes we could look for them here
        
        return {
            "answer": answer,
            "user_message_id": user_message.id,
            "assistant_message_id": assistant_message.id,
            "used_chat_ids": used_chat_ids,
            "pdf_sources": pdf_sources,
            "context": "Context was managed dynamically by the agent."
        }
        
    except Exception as e:
        logger.error("Error in handle_thread_chat", exc_info=True)
        raise e
