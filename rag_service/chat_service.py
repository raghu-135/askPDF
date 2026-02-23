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
from models import (
    DEFAULT_TOKEN_BUDGET,
    DEFAULT_MAX_ITERATIONS,
    RATIO_SEMANTIC_MEMORY,
    CHARS_PER_TOKEN,
)
from database import create_message, get_recent_messages, update_message_context_compact, MessageRole
from reasoning import normalize_ai_response

logger = logging.getLogger(__name__)


async def _build_recent_history_messages(
    thread_id: str,
    context_window: int,
) -> List[Any]:
    """
    Load a compact recent transcript so follow-up questions remain grounded
    even when the model skips history-related tool calls.
    """
    safe_window = max(DEFAULT_TOKEN_BUDGET, int(context_window or DEFAULT_TOKEN_BUDGET))
    # Reuse the semantic-memory allocation ratio as the direct recent-history budget.
    history_budget_chars = int(safe_window * RATIO_SEMANTIC_MEMORY * CHARS_PER_TOKEN)

    # Fetch a ratio-scaled candidate pool, then pack newest->oldest within budget.
    candidate_limit = max(1, int((safe_window * RATIO_SEMANTIC_MEMORY) / max(1, DEFAULT_MAX_ITERATIONS)))
    recent = await get_recent_messages(thread_id, limit=candidate_limit)

    packed_reversed: List[Any] = []
    used_chars = 0

    for msg in reversed(recent):
        if msg.role == MessageRole.USER:
            text = (msg.content or "").strip()
            message_obj = HumanMessage
        elif msg.role == MessageRole.ASSISTANT:
            text = (msg.context_compact or msg.content or "").strip()
            message_obj = AIMessage
        else:
            continue

        if not text:
            continue

        text_chars = len(text)
        if packed_reversed and used_chars + text_chars > history_budget_chars:
            break

        packed_reversed.append(message_obj(content=text))
        used_chars += text_chars

    return list(reversed(packed_reversed))


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
        "system_role": "",
        "tool_instructions": {},
        "custom_instructions": "",
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
    normalized = normalize_ai_response(messages[-1] if messages else None)
    answer = normalized["answer"] or "Error"
    return {
        "answer": answer,
        "reasoning": normalized["reasoning"],
        "reasoning_available": normalized["reasoning_available"],
        "reasoning_format": normalized["reasoning_format"],
        "context": "Legacy context retrieval unsupported",
    }


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
    max_iterations = getattr(req, 'max_iterations', None) or DEFAULT_MAX_ITERATIONS
    system_role = getattr(req, 'system_role_override', "") or ""
    tool_instructions = getattr(req, 'tool_instructions_override', None) or {}
    custom_instructions = getattr(req, 'custom_instructions_override', "") or ""
    
    try:
        from agent import app as agent_app, AgentState
        
        recent_history_messages = await _build_recent_history_messages(
            thread_id=thread_id,
            context_window=context_window,
        )
        initial_state = {
            "messages": recent_history_messages + [HumanMessage(content=question)],
            "thread_id": thread_id,
            "llm_model": llm_model,
            "embedding_model": embed_model,
            "context_window": context_window,
            "use_web_search": use_web_search,
            "pdf_sources": [],
            "used_chat_ids": [],
            "clarification_options": None,
            "iteration_count": 0,
            "max_iterations": max_iterations,
            "system_role": system_role,
            "tool_instructions": tool_instructions,
            "custom_instructions": custom_instructions,
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
        normalized = normalize_ai_response(final_messages[-1] if final_messages else None)
        answer = normalized["answer"] or "Error processing request."
        pdf_sources = result.get("pdf_sources", [])
        used_chat_ids = result.get("used_chat_ids", [])
        clarification_options = result.get("clarification_options", None)
        
        if clarification_options:
            answer = f"I need a bit more clarification. Did you mean:\n" + "\n".join([f"- {opt}" for opt in clarification_options])
            normalized = {
                "reasoning": "",
                "reasoning_available": False,
                "reasoning_format": "none",
            }
        
        # Store messages in database
        user_message = await create_message(
            thread_id=thread_id,
            role=MessageRole.USER,
            content=question
        )
        
        assistant_message = await create_message(
            thread_id=thread_id,
            role=MessageRole.ASSISTANT,
            content=answer,
            reasoning=normalized["reasoning"],
            reasoning_available=normalized["reasoning_available"],
            reasoning_format=normalized["reasoning_format"],
        )
        
        # Index in semantic memory if not a clarification
        if not clarification_options:
            indexing_result = await index_chat_memory_for_thread(
                thread_id=thread_id,
                message_id=assistant_message.id,
                question=question,
                answer=answer,
                embedding_model_name=embed_model,
                llm_name=llm_model,
                context_window=context_window
            )
            compact_text = indexing_result.get("memory_compact_text") if isinstance(indexing_result, dict) else None
            if compact_text:
                await update_message_context_compact(assistant_message.id, compact_text)
        
        return {
            "answer": answer,
            "user_message_id": user_message.id,
            "assistant_message_id": assistant_message.id,
            "used_chat_ids": used_chat_ids,
            "pdf_sources": pdf_sources,
            "clarification_options": clarification_options,
            "reasoning": normalized["reasoning"],
            "reasoning_available": normalized["reasoning_available"],
            "reasoning_format": normalized["reasoning_format"],
            "context": "Context retrieved dynamically by LangGraph Orchestrator tool calls."
        }
        
    except Exception as e:
        logger.error("Error in handle_thread_chat", exc_info=True)
        raise e
