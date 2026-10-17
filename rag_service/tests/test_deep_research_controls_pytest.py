import pytest

from app.services.agent_task_budgets import capped_tranche_increment, initial_budget_state, normalize_budget_state
from app.tools.contracts import InspectDocumentRequest


def test_capped_tranche_increment_stops_at_limit():
    state = initial_budget_state({"max_model_calls": 2})
    state["tranche_usage"]["model_calls"] = 2
    assert capped_tranche_increment(state, "model_calls", 5) == 0
    state["tranche_usage"]["model_calls"] = 1
    assert capped_tranche_increment(state, "model_calls", 5) == 1


def test_inspect_document_request_coerces_int_cursor():
    req = InspectDocumentRequest(document_id="doc-1", cursor=0)
    assert isinstance(req.cursor, str)
    assert req.cursor
