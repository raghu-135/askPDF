from __future__ import annotations

from langgraph_runtime.workflows.answer_numeric_grounding import (
    numeric_grounding_violations,
    extract_numeric_literals,
)


def test_extract_numeric_literals_preserves_scientific_forms():
    text = "Base cost 3.3 x 10^18 and big 2.3 x 10^19 FLOPs."
    literals = extract_numeric_literals(text)
    assert "3.3 x 10^18" in literals
    assert "2.3 x 10^19" in literals


def test_numeric_grounding_violation_for_wrong_magnitude_reformat():
    evidence = "Table 2 EN-FR big-model training cost is 2.3 x 10^19 FLOPs; base is 3.3 x 10^18."
    answer = "The EN-FR big model cost is about 3.3e+23 FLOPs."
    assert numeric_grounding_violations(answer, evidence) == ["3.3e+23"]


def test_numeric_grounding_accepts_verbatim_evidence_literal():
    evidence = "Table 2 EN-FR big-model training cost is 2.3 x 10^19 FLOPs."
    answer = "The big-model EN-FR cost is 2.3 x 10^19 FLOPs."
    assert numeric_grounding_violations(answer, evidence) == []
