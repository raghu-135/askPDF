from app.rag.retrieval import clip_retrieval_hit


def test_clip_retrieval_hit_preserves_tail_for_long_table_rows():
    tail = "Transformer (big) 28.4"
    filler = "x" * 2000
    text = filler + tail
    clipped, truncated = clip_retrieval_hit(text, hit_limit=200)
    assert truncated is True
    assert tail in clipped
    assert clipped.startswith("x")
