"""
chat_service.py - Business logic for chat endpoint in RAG Service

This module provides the chat handling logic for the /chat endpoint.
"""

from langchain_core.messages import AIMessage, HumanMessage
from agent import app as agent_app

async def handle_chat(req):
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
    # Return answer and context (could also return web_context if desired)
    return {"answer": result["answer"], "context": result.get("context", "")}
