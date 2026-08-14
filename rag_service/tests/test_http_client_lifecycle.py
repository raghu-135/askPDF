import pytest


@pytest.mark.asyncio
async def test_owned_http_clients_are_closed():
    from app.http_clients import close_http_clients, register_owned_client

    class FakeClient:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    client = register_owned_client(FakeClient())
    await close_http_clients()
    assert client.closed is True


@pytest.mark.asyncio
async def test_lazy_http_client_is_registered_and_closed(monkeypatch):
    from app.http_clients import close_http_clients, get_http_client

    monkeypatch.setenv("HTTP_CLIENT_TIMEOUT_SECONDS", "1")
    client = get_http_client("lifecycle-test")
    assert client.is_closed is False
    await close_http_clients()
    assert client.is_closed is True


@pytest.mark.asyncio
async def test_repeated_http_shutdown_is_idempotent():
    from app.http_clients import close_http_clients

    await close_http_clients()
    await close_http_clients()


@pytest.mark.asyncio
async def test_http_startup_fills_missing_named_clients(monkeypatch):
    import app.http_clients as http_clients

    await http_clients.close_http_clients()
    monkeypatch.setenv("HTTP_CLIENT_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("MCP_REQUEST_TIMEOUT_SECONDS", "1")
    http_clients._clients["llm"] = http_clients.httpx.AsyncClient(timeout=1)
    try:
        await http_clients.init_http_clients()
        assert {"default", "llm", "embeddings", "providers", "mcp"}.issubset(http_clients._clients)
        assert http_clients._clients["llm"].is_closed is False
    finally:
        await http_clients.close_http_clients()


@pytest.mark.parametrize("value", ["not-a-number", "0", "-1", "nan", "inf", "-inf"])
def test_http_timeout_rejects_non_finite_or_non_positive_values(monkeypatch, value):
    import app.http_clients as http_clients

    monkeypatch.setenv("HTTP_CLIENT_TIMEOUT_SECONDS", value)
    with pytest.raises(RuntimeError, match="HTTP_CLIENT_TIMEOUT_SECONDS"):
        http_clients._timeout()


@pytest.mark.asyncio
async def test_application_startup_failure_closes_http_clients(monkeypatch):
    import main as application

    initialized = False
    closed = False

    async def fake_init_http_clients():
        nonlocal initialized
        initialized = True

    async def failing_init_db():
        raise RuntimeError("database startup failed")

    async def fake_close_http_clients():
        nonlocal closed
        closed = True

    async def fake_close_db():
        return None

    async def fake_shutdown_memory_repairs():
        return None

    monkeypatch.setattr(application, "init_http_clients", fake_init_http_clients)
    monkeypatch.setattr(application, "init_db", failing_init_db)
    monkeypatch.setattr(application, "close_http_clients", fake_close_http_clients)
    monkeypatch.setattr(application, "close_db", fake_close_db)
    monkeypatch.setattr(application, "close_vector_db", lambda: None)
    monkeypatch.setattr(application, "shutdown_memory_repairs", fake_shutdown_memory_repairs)

    with pytest.raises(RuntimeError, match="database startup failed"):
        async with application.lifespan(application.app):
            raise AssertionError("startup failure should prevent entering lifespan")

    assert initialized is True
    assert closed is True
