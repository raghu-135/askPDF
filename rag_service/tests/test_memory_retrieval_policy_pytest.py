from types import SimpleNamespace

from app.db.vector.helpers import _score, _score_type
from app.services.memory_retrieval_policy import (
    compute_memory_retrieval_budget,
    memory_is_applicable,
    pack_memory_results,
    token_similarity,
)


def test_memory_budget_is_a_ratio_of_the_context_window():
    import math

    from app.models.llm_server_client import CHARS_PER_TOKEN
    from app.models.memory_limits import MAX_CONTEXT_WINDOW_TOKENS, MAX_MEMORY_CONTEXT_CHARS
    from app.services.memory_retrieval_policy import AVERAGE_MEMORY_CHARS
    from runtime_protocol.context_budget import load_context_budget_settings

    settings = load_context_budget_settings()
    previous = None
    for context_window in (4096, 16384, 32768, 131072, 1_000_000):
        budget = compute_memory_retrieval_budget(context_window)
        usable = int(context_window * settings.usable_ratio * CHARS_PER_TOKEN)
        assert budget["char_budget"] == max(1, int(usable * budget["ratio"]))
        assert budget["candidate_limit"] == max(
            1, math.ceil(budget["char_budget"] / AVERAGE_MEMORY_CHARS) * 2
        )
        if previous is not None:
            assert budget["char_budget"] > previous
        previous = budget["char_budget"]

    standard = compute_memory_retrieval_budget(1_000_000)
    expanded = compute_memory_retrieval_budget(1_000_000, expanded=True)
    assert expanded["char_budget"] > standard["char_budget"]
    assert expanded["ratio"] == standard["ratio"] * 2
    ceiling = compute_memory_retrieval_budget(MAX_CONTEXT_WINDOW_TOKENS, expanded=True)
    assert ceiling["char_budget"] == MAX_MEMORY_CONTEXT_CHARS


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
