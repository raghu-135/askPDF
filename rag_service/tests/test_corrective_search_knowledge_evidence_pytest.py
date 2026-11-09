from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.tools.contracts import SearchKnowledgeRequest
from app.tools.retrieval_knowledge import search_knowledge
from tests.test_first_party_tool_contracts import _patch_ready_manifest


@pytest.mark.asyncio
async def test_search_knowledge_emits_document_evidence_segments(monkeypatch):
    _patch_ready_manifest(monkeypatch)
    fake_db = SimpleNamespace(
        search_knowledge_sources=AsyncMock(
            return_value=[
                {
                    "file_hash": "abcdefabcdefabcdefabcdefabcdefab",
                    "chunk_id": "c-table2",
                    "source_id": "src-table2",
                    "score": 0.92,
                    "text": "legacy",
                    "manifest_id": "manifest-file-1",
                    "generation": "generation-1",
                    "metadata": {
                        "body_text": "Base model training cost is 3.3 x 10^18 FLOPs (Table 2).",
                        "pages": [7],
                        "extraction_fingerprint": "extract-1",
                        "chunking_fingerprint": "chunk-1",
                    },
                },
            ]
        ),
    )

    class Services:
        async def embed(self, _model, _query):
            return [0.1, 0.2, 0.3]

        def vector_db(self):
            return fake_db

        async def document_lookup(self, _thread_id):
            return {"abcdefabcdefabcdefabcdefabcdefab": {"file_name": "attention.pdf"}}

        async def rerank(self, _query, chunks):
            return chunks

    raw = await search_knowledge(
        SearchKnowledgeRequest(query="training cost Table 2", level="chunk", max_results=5),
        SimpleNamespace(
            thread_id="thread-1",
            embedding_model="embed-1",
            use_reranker=False,
            caller_node="retrieval_worker",
        ),
        services=Services(),
    )
    artifacts = raw.artifacts
    assert artifacts["evidence_segments"]
    segment = artifacts["evidence_segments"][0]
    assert "3.3" in segment["content"]
    assert segment["source_id"].startswith("doc:abcdefabcdefabcdefabcdefabcdefab:")
