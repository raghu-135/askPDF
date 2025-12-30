from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from models import get_llm, get_embedding_model
from vectordb.qdrant import QdrantAdapter

class AgentState(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    llm_model: str
    embedding_model: str
    context: str
    answer: str

def retrieve(state: AgentState):
    question = state["question"]
    emb_model_name = state["embedding_model"]
    
    # Get embedding for query
    embed_model = get_embedding_model(emb_model_name)
    query_vector = embed_model.embed_query(question)
    
    # Search Qdrant
    base_model_name = emb_model_name.split(":")[0]
    safe_name = base_model_name.replace("-", "_").replace(".", "_").replace("/", "_")
    collection_name = f"rag_{safe_name}"
    
    db = QdrantAdapter() # In future, can be dynamic
    # Search is async in adapter, but here we might run sync or use ainvoke if graph is async
    # For simplicity, let's assume we run this in sync or bridge it. 
    # Actually QdrantAdapter search was defined as async. 
    # LangGraph nodes can be async.
    import asyncio
    # We'll rely on the runner executing this async
    
    # Note: If this function is not async, we use `asyncio.run`, but inside async framework (fastapi) that's bad.
    # Better to make this node async.
    return {"context": ""} # Placeholder, we need to return awaitable if defined as async

async def retrieve_node(state: AgentState):
    question = state["question"]
    emb_model_name = state["embedding_model"]
    
    embed_model = get_embedding_model(emb_model_name)
    # embed_query is usually sync in langchain integration unless using aembed_query
    query_vector = await embed_model.aembed_query(question)
    
    base_model_name = emb_model_name.split(":")[0]
    safe_name = base_model_name.replace("-", "_").replace(".", "_").replace("/", "_")
    collection_name = f"rag_{safe_name}"
    
    db = QdrantAdapter()
    results = await db.search(collection_name, query_vector, limit=5)
    
    context = "\n\n".join([res["text"] for res in results])
    return {"context": context}

async def generate_node(state: AgentState):
    question = state["question"]
    context = state["context"]
    llm_name = state["llm_model"]
    
    llm = get_llm(llm_name)
    
    messages = [
        SystemMessage(content="You are a helpful assistant. Use the following context to answer the user's question. If you don't know, say so."),
        SystemMessage(content=f"Context:\n{context}"),
    ]
    # Add history
    messages.extend(state.get("chat_history", []))
    messages.append(HumanMessage(content=question))
    
    response = await llm.ainvoke(messages)
    return {"answer": response.content}

# Define Graph
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
