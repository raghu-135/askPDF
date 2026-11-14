import pytest


@pytest.mark.asyncio
async def test_generate_embeddings_local_uses_single_pass(monkeypatch):
    from app.rag import indexer

    calls: list[int] = []

    class _Client:
        async def aembed_documents(self, texts):
            calls.append(len(texts))
            return [[float(len(texts))] for _ in texts]

    monkeypatch.setattr(indexer, "get_embedding_model", lambda _model: _Client())
    monkeypatch.setattr(
        "app.models.llm_server_client.should_use_local_embeddings",
        lambda _model: True,
    )

    async def _invoke(func, *args, **kwargs):
        return await func(*args, **kwargs)

    monkeypatch.setattr("app.models.retry.invoke_with_retry", _invoke)

    chunks = [f"chunk-{i}" for i in range(250)]
    vectors = await indexer.generate_embeddings(chunks, "BAAI/bge-m3")

    assert calls == [250]
    assert len(vectors) == 250


@pytest.mark.asyncio
async def test_generate_embeddings_remote_batches(monkeypatch):
    from app.rag import indexer

    calls: list[int] = []

    class _Client:
        async def aembed_documents(self, texts):
            calls.append(len(texts))
            return [[1.0] for _ in texts]

    monkeypatch.setattr(indexer, "get_embedding_model", lambda _model: _Client())
    monkeypatch.setattr(
        "app.models.llm_server_client.should_use_local_embeddings",
        lambda _model: False,
    )

    async def _invoke(func, *args, **kwargs):
        return await func(*args, **kwargs)

    monkeypatch.setattr("app.models.retry.invoke_with_retry", _invoke)

    chunks = [f"chunk-{i}" for i in range(250)]
    vectors = await indexer.generate_embeddings(chunks, "openai/text-embedding-3-small")

    assert calls == [100, 100, 50]
    assert len(vectors) == 250