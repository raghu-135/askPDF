from types import SimpleNamespace

from app.prompts.loaders import load_prompt
from app.models.memory_manager_input_budget import compute_memory_manager_input_budget
from app.services.memory_review_service import (
    _review_override_edges,
    _visible_review_neighbors,
    build_review_memory_search_query,
)


def test_review_search_query_is_bounded_and_represents_every_turn():
    turns = [{
        "question": f"Question {index} " + ("q" * 2000),
        "answer": f"Answer {index} " + ("a" * 2000),
    } for index in range(20)]

    query = build_review_memory_search_query(
        [{"role": "user", "content": "Review these completed turns."}],
        {"turns": turns},
    )

    assert len(query) <= 12000
    assert "Review these completed turns." in query
    assert all(f"Question {index}" in query for index in range(20))


def test_curator_budget_scales_with_selected_context_window():
    small = compute_memory_manager_input_budget(2048)
    large = compute_memory_manager_input_budget(40000)

    assert large["transcript_chars"] > small["transcript_chars"]
    assert large["review_context_chars"] > small["review_context_chars"]
    assert large["memory_context_chars"] > small["memory_context_chars"]
    assert large["memory_result_limit"] > small["memory_result_limit"]


def test_review_neighbors_exclude_override_targets_outside_visible_context():
    visible = {"anchor": object(), "visible-hit": object(), "visible-override": object()}
    edges = [
        SimpleNamespace(overriding_memory_id="anchor", overridden_memory_id="visible-override"),
        SimpleNamespace(overriding_memory_id="anchor", overridden_memory_id="other-project-memory"),
    ]

    assert _visible_review_neighbors(
        "anchor",
        ["visible-hit", "other-project-hit", "visible-hit"],
        edges,
        visible,
    ) == ["visible-hit", "visible-override"]


def test_review_neighbors_exclude_self_and_preserve_incoming_visible_override():
    visible = {"anchor": object(), "narrower-memory": object()}
    edges = [
        SimpleNamespace(overriding_memory_id="narrower-memory", overridden_memory_id="anchor"),
    ]

    assert _visible_review_neighbors(
        "anchor",
        ["anchor"],
        edges,
        visible,
    ) == ["narrower-memory"]


def test_review_override_edges_include_only_relationships_inside_candidate_group():
    edges = [
        SimpleNamespace(overriding_memory_id="thread", overridden_memory_id="global"),
        SimpleNamespace(overriding_memory_id="other-thread", overridden_memory_id="global"),
    ]

    assert _review_override_edges(["thread", "global"], edges) == [{
        "overriding_memory_id": "thread",
        "overridden_memory_id": "global",
    }]


def test_memory_review_prompt_teaches_scope_precedence_and_existing_overrides():
    prompt = load_prompt("memory_manager/system.md")
    prompt_flat = " ".join(prompt.split())

    assert "clean up existing stored memories and relationships" in prompt_flat
    assert "Do not mine new facts from the conversation or invent unrelated new memories" in prompt_flat
    assert "Same-scope conflicts are not hierarchy conflicts" in prompt_flat
    assert "do not propose Thread overrides for two Global memories" in prompt_flat
    assert "Override in the narrower scope (recommended)" in prompt
    assert "override_edges" in prompt
    assert "Put the recommended contextual override first" in prompt
    assert "Update the broader memory" in prompt
    assert "changes behavior for every project or thread" in prompt
    assert "Unrelated and additive memories require no operation" in prompt
    assert "Do not create unrelated new memories from reviewer inference" in prompt_flat
    assert 'mode` is `conversation_review' in prompt
    assert 'only `create` intents with `scope_type="thread"`' in prompt
    assert "Do not save incidental context, episode summaries, temporary task state, inferred personal facts" in prompt_flat
