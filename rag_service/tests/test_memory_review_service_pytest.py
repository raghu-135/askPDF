from types import SimpleNamespace

from app.services.memory_review_service import _visible_review_neighbors


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
