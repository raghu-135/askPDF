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


def test_inspect_document_request_coerces_string_digit_cursor():
    req = InspectDocumentRequest(document_id="doc-1", cursor="0")
    assert isinstance(req.cursor, str)
    assert req.cursor


def test_read_context_request_coerces_long_expansion():
    from app.tools.contracts import ReadContextRequest

    req = ReadContextRequest(source_id="src_abc", expansion="long")
    assert req.expansion == "section"


def test_read_context_request_rejects_parameter_name_cursors():
    from app.tools.contracts import ReadContextRequest

    for cursor in ("page_size", "start", "0", "section", ""):
        req = ReadContextRequest(source_id="src_abc", cursor=cursor)
        assert req.cursor is None


def test_read_context_request_accepts_v2_continuation_token():
    from app.tools.contracts import ReadContextRequest
    from app.tools.retrieval_knowledge import _context_cursor

    token = _context_cursor(
        manifest_id="manifest-1",
        generation="generation-1",
        source_version="source-1",
        anchor_chunk_id="chunk-1",
        expansion="section",
        token_budget=512,
        tokenizer_fingerprint="tokenizer-1",
        segment_index=1,
    )
    req = ReadContextRequest(source_id="src_abc", cursor=token)
    assert req.cursor == token
