"""
chat_service.py - Business logic for chat endpoints in RAG Service

This module provides:
- Legacy chat handling (handle_chat)
- Thread-based chat with semantic memory (handle_thread_chat)
- Token budgeting for context management
- Dual-search (PDF chunks + chat memory)
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent import app as agent_app, invoke_with_retry
from models import get_llm, get_embedding_model
from vectordb.qdrant import QdrantAdapter
from database import (
    create_message, get_recent_messages, get_thread,
    MessageRole
)

# Token budget configuration
DEFAULT_TOKEN_BUDGET = 4000  # Conservative estimate for context
PDF_CHUNK_PRIORITY = 3  # Highest priority
RECENT_HISTORY_PRIORITY = 2  # Second priority
SEMANTIC_MEMORY_PRIORITY = 1  # Third priority

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
    Budget the context to fit within token limits.
    Priority: Question (mandatory) > PDF chunks > Recent history > Recalled memories
    
    Returns filtered lists that fit within budget.
    """
    used_tokens = estimate_tokens(question)
    
    # Reserve some tokens for the answer
    available_tokens = max_tokens - 500  # Reserve 500 tokens for response
    
    selected_pdf_chunks = []
    selected_messages = []
    selected_memories = []
    
    # 1. Add PDF chunks (highest priority)
    for chunk in pdf_chunks:
        chunk_tokens = estimate_tokens(chunk.get("text", ""))
        if used_tokens + chunk_tokens < available_tokens:
            selected_pdf_chunks.append(chunk)
            used_tokens += chunk_tokens
        else:
            break
    
    # 2. Add recent messages (second priority)
    for msg in recent_messages:
        msg_tokens = estimate_tokens(msg.content)
        if used_tokens + msg_tokens < available_tokens:
            selected_messages.append(msg)
            used_tokens += msg_tokens
        else:
            break
    
    # 3. Add recalled memories (third priority, trim first if over budget)
    for memory in recalled_memories:
        memory_tokens = estimate_tokens(memory.get("text", ""))
        if used_tokens + memory_tokens < available_tokens:
            selected_memories.append(memory)
            used_tokens += memory_tokens
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
    
    try:
        # 1. Embed the question
        embed_model_client = get_embedding_model(embed_model)
        query_vector = await invoke_with_retry(embed_model_client.aembed_query, question)
        
        # 2. Search PDF chunks
        pdf_chunks = await db.search_pdf_chunks(
            thread_id=thread_id,
            query_vector=query_vector,
            limit=5
        )
        
        # 3. Search chat memory (excluding current conversation)
        recent_msgs = await get_recent_messages(thread_id, limit=4)
        recent_message_ids = [m.id for m in recent_msgs]
        
        recalled_memories = await db.search_chat_memory(
            thread_id=thread_id,
            query_vector=query_vector,
            limit=3,
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
            question=question
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
            try:
                from langchain_community.tools import DuckDuckGoSearchRun
                search_tool = DuckDuckGoSearchRun()
                web_results = await asyncio.to_thread(search_tool.invoke, question)
                web_context = f"\n\nWeb Search Results:\n{web_results}"
                full_context += web_context
            except Exception as e:
                print(f"Web search failed: {e}", flush=True)
        
        # 8. Generate response
        llm = get_llm(llm_model)
        
        messages = [
            SystemMessage(content=(
                "You are a helpful assistant. Use the provided context to answer the user's question. "
                "The context includes PDF content and may include relevant past conversations. "
                "If you reference past conversations, acknowledge it naturally. "
                "If you cannot answer from the context, say so."
            )),
            SystemMessage(content=f"Context:\n{full_context}"),
        ]
        
        # Add recent history
        messages.extend(selected_history)
        messages.append(HumanMessage(content=question))
        
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
        import traceback
        traceback.print_exc()
        raise e
