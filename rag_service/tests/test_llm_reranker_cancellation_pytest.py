import pytest

from app.models.llm_server_client import LocalRerankerWrapper


@pytest.mark.asyncio
async def test_local_reranker_honors_cancellation_between_batches():
    calls = {"predict": 0}

    class FakeModel:
        def predict(self, pairs):
            calls["predict"] += 1
            return [0.5] * len(pairs)

    wrapper = LocalRerankerWrapper(FakeModel())

    def checker():
        return True

    passages = [f"passage-{index}" for index in range(40)]
    scores = await wrapper.ascore("query", passages, cancellation_checker=checker)
    assert len(scores) == len(passages)
    assert calls["predict"] == 0
