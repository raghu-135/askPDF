from runtime_protocol.evidence import inherited_evidence_packets

_TEST_CONTEXT_BUDGET = {
    "usable_ratio": 0.8,
    "recent_ratio": 0.22,
    "semantic_ratio": 0.12,
    "durable_memory_ratio": 0.06,
    "document_ratio": 0.28,
    "memory_expanded_ratio": 0.12,
    "evidence_ratio": 0.45,
    "draft_ratio": 0.25,
    "plan_section_ratio": 0.20,
    "plan_total_ratio": 0.50,
    "plan_memory_ratio": 0.12,
}


def test_inherited_document_packet_uses_inventory_when_prefetch_evidence_empty():
    bundle = {
        "document_evidence_text": "",
        "documents": [
            {
                "file_name": "Raghunandan_Reddy_Resume_plain.pdf",
                "file_hash": "abc123",
                "chunk_count": 7,
            },
        ],
    }
    packets = inherited_evidence_packets(
        bundle,
        profile_id="document_researcher",
        context_window=30_000,
        context_budget=_TEST_CONTEXT_BUDGET,
    )
    assert len(packets) == 1
    packet = packets[0]
    assert packet.available
    assert "Raghunandan_Reddy_Resume_plain.pdf" in packet.content
    assert "thread_inventory_metadata_only" in packet.warnings


def test_inherited_document_packet_prefers_thread_shape_text():
    bundle = {
        "document_evidence_text": "",
        "thread_shape_text": "[THREAD SHAPE]\nDocuments   : 1 source(s)\n  1. file_name=resume.pdf",
        "documents": [{"file_name": "resume.pdf", "file_hash": "h1", "chunk_count": 3}],
    }
    packets = inherited_evidence_packets(
        bundle,
        profile_id="document_researcher",
        context_window=30_000,
        context_budget=_TEST_CONTEXT_BUDGET,
    )
    assert packets[0].content.startswith("[THREAD SHAPE]")
