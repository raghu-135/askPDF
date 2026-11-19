from app.db.jsonb_utils import postgres_jsonb_safe


def test_postgres_jsonb_safe_roundtrips_nested_dict():
    value = {"text": "café", "n": 1}
    assert postgres_jsonb_safe(value) == value


def test_postgres_jsonb_safe_strips_lone_surrogates():
    dirty = {"text": "bad\ud800pair"}
    cleaned = postgres_jsonb_safe(dirty)
    assert "\ud800" not in cleaned["text"]
    assert postgres_jsonb_safe({"nested": [{"x": "\udc00"}]})["nested"][0]["x"]