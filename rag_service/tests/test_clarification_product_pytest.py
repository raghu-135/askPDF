from app.product_orchestration.clarification import (
    CLARIFICATION_REQUIRED_STATUS,
    is_clarification_terminal_result,
    normalize_clarification_terminal_status,
)


def test_normalize_clarification_status_from_options():
    result = {"clarification_options": ["What is in the PDF?", "What sources exist?"]}
    assert is_clarification_terminal_result(result)
    assert normalize_clarification_terminal_status("completed", result) == CLARIFICATION_REQUIRED_STATUS


def test_normalize_clarification_status_from_legacy_status():
    result = {}
    assert normalize_clarification_terminal_status("clarification", result) == CLARIFICATION_REQUIRED_STATUS