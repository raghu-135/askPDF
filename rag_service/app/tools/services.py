"""Application dependency provider for framework-neutral tool handlers."""

from typing import Any


class DefaultToolServices:
    """Lazy service provider used by MCP handlers and compatibility adapters.

    Imports stay inside methods so importing the neutral tool layer does not
    initialize optional providers or framework-specific runtimes.
    """

    async def get_thread_shape(self, thread_id: str) -> dict[str, Any]:
        from app.db import get_thread_shape
        return await get_thread_shape(thread_id)

    def vector_db(self) -> Any:
        from app.db.vector import get_vector_db
        return get_vector_db()

    async def document_lookup(self, thread_id: str) -> dict[str, Any]:
        from app.rag.retrieval import get_document_metadata_lookup
        return await get_document_metadata_lookup(thread_id)

    async def embed(self, model: str, query: str) -> list[float]:
        from app.models.llm_server_client import embed_query
        return await embed_query(model, query)

    async def rerank(self, query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from app.rag.retrieval import rerank_document_chunks
        return await rerank_document_chunks(query, chunks)

    async def semantic_history(self, **kwargs: Any) -> Any:
        from app.rag.retrieval import fetch_semantic_history
        return await fetch_semantic_history(**kwargs)

    def evidence_segment(self, **kwargs: Any) -> dict[str, Any] | None:
        from app.agent.evidence_contract import evidence_segment
        return evidence_segment(**kwargs)


def get_tool_services() -> DefaultToolServices:
    return DefaultToolServices()
