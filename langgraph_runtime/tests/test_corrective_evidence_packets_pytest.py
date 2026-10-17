from langgraph_runtime.workflows.corrective_contracts import CORRECTIVE_WORKFLOW_ID
from langgraph_runtime.workflows.evidence import corrective_evidence_packets


def test_corrective_evidence_packets_accepts_multiple_sources_when_provenance_omitted():
    state = {
        "workflow_id": CORRECTIVE_WORKFLOW_ID,
        "corrective_policy": {},
        "evidence_packets": [
            {
                "id": "packet-1",
                "source_ids": [
                    "doc:abcdefabcdefabcdefabcdefabcdefab:0",
                    "doc:abcdefabcdefabcdefabcdefabcdefab:1",
                ],
                "content": "Shared document evidence",
            },
        ],
        "evidence_assessments": [
            {
                "packet_id": "packet-1",
                "relevant": True,
                "confidence": 0.9,
                "instruction_injection_risk": False,
                "coverage": ["claim-a"],
            },
        ],
    }
    selected = corrective_evidence_packets(state)
    assert len(selected) == 1
    assert len(selected[0]["source_ids"]) == 2
