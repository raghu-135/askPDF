from types import SimpleNamespace

from app.db.vector.helpers import _score, _score_type
from app.services.memory_retrieval_policy import (
    compute_memory_retrieval_budget,
    memory_is_applicable,
    pack_memory_results,
    token_similarity,
)


def test_memory_budget_scales_and_caps_across_context_windows():
    expected = {
        4096: (800, 5),
        16384: (3145, 16),
        32768: (6291, 20),
        131072: (8000, 20),
        1_000_000: (8000, 20),
    }
    for context_window, (chars, candidates) in expected.items():
        budget = compute_memory_retrieval_budget(context_window)
        assert budget["char_budget"] == chars
        assert budget["candidate_limit"] == candidates

    expanded = compute_memory_retrieval_budget(1_000_000, expanded=True)
    assert expanded["char_budget"] == 16000
    assert expanded["candidate_limit"] == 40


def test_memory_applicability_uses_attributes_without_an_llm_call():
    assert memory_is_applicable({"kind": "preference", "applicability": ["all_answers"]}, "hello")
    assert memory_is_applicable({"kind": "instruction", "applicability": ["code"]}, "show Python code")
    assert not memory_is_applicable({"kind": "instruction", "applicability": ["code"]}, "draft an email")


def test_memory_packing_preserves_short_items_and_respects_budget():
    packed, used = pack_memory_results([
        {"id": "one", "content": "short preference"},
        {"id": "two", "content": "x" * 1200},
    ], 400)
    assert packed[0]["excerpt"] == "short preference"
    assert used <= 400
    assert sum(len(item["excerpt"]) for item in packed) == used


def test_near_duplicate_similarity_is_token_based():
    assert token_similarity(
        "Prefer concise TypeScript examples with clear names",
        "Prefer clear, concise TypeScript examples with names",
    ) >= 0.85
    assert token_similarity("Prefer Python", "Use detailed tables") < 0.85


def test_distance_fallback_is_normalized_to_higher_is_better_similarity():
    exact = SimpleNamespace(metadata=SimpleNamespace(score=None, distance=0.0))
    farther = SimpleNamespace(metadata=SimpleNamespace(score=None, distance=1.0))
    hybrid = SimpleNamespace(metadata=SimpleNamespace(score=0.42, distance=0.2))

    assert _score(exact) == 1.0
    assert _score(farther) == 0.5
    assert _score(exact) > _score(farther)
    assert _score_type(farther) == "distance_similarity"
    assert _score_type(hybrid) == "hybrid_score"
