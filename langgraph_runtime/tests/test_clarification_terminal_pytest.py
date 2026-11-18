from langgraph_runtime.adapter import _result_from_graph
from langgraph_runtime.workflows.clarification_terminal import (
    CLARIFICATION_REQUIRED_STATUS,
    clarification_terminal_state_update,
)
from runtime_protocol.contracts import AgentRuntimeResult


def test_clarification_terminal_state_clears_answer():
    update = clarification_terminal_state_update(
        ["What is in the uploaded PDF?", "What web sources are available?"]
    )
    assert update["final_answer"] == ""
    assert update["status"] == CLARIFICATION_REQUIRED_STATUS
    assert len(update["clarification_options"]) == 2


def test_result_from_graph_maps_clarification_status():
    raw = {
        "clarification_options": ["A", "B"],
        "final_answer": "",
        "status": CLARIFICATION_REQUIRED_STATUS,
    }
    result = _result_from_graph(raw)
    assert isinstance(result, AgentRuntimeResult)
    assert result.status == CLARIFICATION_REQUIRED_STATUS
    assert result.clarification == {"options": ["A", "B"]}
    assert result.output.get("answer") in ("", None)