"""
chat_service.py - Business logic for chat endpoints in RAG Service

This module provides:
- Legacy chat handling (handle_chat)
- Thread-based chat with semantic memory (handle_thread_chat)
"""

import asyncio
import logging
from typing import List, Dict, Any

from langchain_core.messages import AIMessage, HumanMessage

from rag import index_chat_memory_for_thread
from models import DEFAULT_TOKEN_BUDGET, DEFAULT_MAX_ITERATIONS
from database import create_message, MessageRole

logger = logging.getLogger(__name__)


async def handle_chat(req) -> Dict[str, Any]:
    """
    Legacy chat handler for backward compatibility.
    Uses collection-based retrieval without semantic memory.
    """
    from agent import app as agent_app
    chat_history = []
    for msg in req.history:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))

    inputs = {
        "messages": chat_history + [HumanMessage(content=req.question)],
        "question": req.question,
        "llm_model": req.llm_model,
        "embedding_model": req.embedding_model,
        "collection_name": req.collection_name,
        "use_web_search": req.use_web_search,
        "context": "",
        "web_context": "",
        "answer": "",
        "iteration_count": 0,
        "max_iterations": getattr(req, 'max_iterations', DEFAULT_MAX_ITERATIONS),
    }
    
    # Provide a dummy config for backward compatibility
    config = {
        "configurable": {
            "thread_id": "legacy_thread",
            "embedding_model": req.embedding_model,
            "context_window": 8000
        }
    }
    
    result = await agent_app.ainvoke(inputs, config=config)
    messages = result.get("messages", [])
    answer = messages[-1].content if messages else "Error"
    return {"answer": answer, "context": "Legacy context retrieval unsupported"}


async def handle_thread_chat(
    thread_id: str,
    req,  # ThreadChatRequest
    embed_model: str
) -> Dict[str, Any]:
    """
    Thread-based chat using Orchestrator Agent with dynamic memory, PDF search, and web tools.
    """
    question = req.question
    llm_model = req.llm_model
    use_web_search = getattr(req, 'use_web_search', False)
    context_window = getattr(req, 'context_window', DEFAULT_TOKEN_BUDGET)
    
    try:
        from agent import app as agent_app, AgentState
        
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "thread_id": thread_id,
            "llm_model": llm_model,
            "embedding_model": embed_model,
            "context_window": context_window,
            "use_web_search": use_web_search,
            "pdf_sources": [],
            "used_chat_ids": [],
            "clarification_options": None,
            "iteration_count": 0,
            "max_iterations": getattr(req, 'max_iterations', DEFAULT_MAX_ITERATIONS)
        }
        
        config = {
            "configurable": {
                "thread_id": thread_id,
                "embedding_model": embed_model,
                "context_window": context_window
            }
        }
        
        logger.info(f"Invoking Orchestrator Agent for thread {thread_id}")
        result = await agent_app.ainvoke(initial_state, config=config)
        
        final_messages = result.get("messages", [])
        answer = final_messages[-1].content if final_messages else "Error processing request."
        pdf_sources = result.get("pdf_sources", [])
        used_chat_ids = result.get("used_chat_ids", [])
        clarification_options = result.get("clarification_options", None)
        
        if clarification_options:
            answer = f"I need a bit more clarification. Did you mean:\n" + "\n".join([f"- {opt}" for opt in clarification_options])
        
        # Store messages in database
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
        
        # Index in semantic memory if not a clarification
        if not clarification_options:
            await index_chat_memory_for_thread(
                thread_id=thread_id,
                message_id=assistant_message.id,
                question=question,
                answer=answer,
                embedding_model_name=embed_model,
                llm_name=llm_model,
                context_window=context_window
            )
        
        return {
            "answer": answer,
            "user_message_id": user_message.id,
            "assistant_message_id": assistant_message.id,
            "used_chat_ids": used_chat_ids,
            "pdf_sources": pdf_sources,
            "clarification_options": clarification_options,
            "context": "Context retrieved dynamically by LangGraph Orchestrator tool calls."
        }
        
    except Exception as e:
        logger.error("Error in handle_thread_chat", exc_info=True)
        raise e
