from __future__ import annotations

from langgraph_runtime.workflows.evidence import final_context_from_state


def test_final_context_merges_prefetch_and_worker_packets():
    state = {
        "workflow_id": "router_rag_agent",
        "pre_fetch_bundle": {
            "documents": [{"file_hash": "a" * 32}],
            "document_evidence_text": "Table 2 lists training costs. Base EN-FR is 3.3 x 10^18 FLOPs.",
        },
        "evidence": "[Document evidence]\nOld worker-only snippet without table.",
        "evidence_packets": [
            {
                "id": "pkt-1",
                "kind": "document",
                "content": "Big model EN-FR training costs 2.3 x 10^19 FLOPs per Table 2.",
                "source_ids": [f"doc:{'a' * 32}:chunk-table"],
            }
        ],
    }
    context, source = final_context_from_state(state)
    assert source == "merged_document_context"
    lowered = context.casefold()
    assert "table 2" in lowered
    assert "3.3" in context
    assert "2.3" in context