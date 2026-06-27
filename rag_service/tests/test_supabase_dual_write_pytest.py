import pytest

from app.db import dual_write, supabase_client


@pytest.fixture
def captured_execute(monkeypatch):
    calls = []

    async def fake_execute(sql, *args):
        calls.append((sql, args))
        return "OK"

    monkeypatch.setenv("RAG_SUPABASE_DUAL_WRITE", "true")
    monkeypatch.setattr(supabase_client, "execute", fake_execute)
    return calls


@pytest.mark.asyncio
async def test_dual_write_disabled_closes_unawaited_coroutine(monkeypatch):
    monkeypatch.setenv("RAG_SUPABASE_DUAL_WRITE", "false")
    closed = False

    class DummyCoro:
        def close(self):
            nonlocal closed
            closed = True

    await dual_write._swallow("disabled", DummyCoro())
    assert closed is True


@pytest.mark.asyncio
async def test_dual_write_failure_is_swallowed(monkeypatch, caplog):
    monkeypatch.setenv("RAG_SUPABASE_DUAL_WRITE", "true")

    async def boom(*args, **kwargs):
        raise RuntimeError("target unavailable")

    await dual_write._swallow("mirror", boom())
    assert "target unavailable" in caplog.text


def test_supabase_url_normalizes_sqlalchemy_asyncpg_scheme():
    url = supabase_client.normalize_db_url("postgresql+asyncpg://user:pass@db:5432/postgres")
    assert url == "postgresql://user:pass@db:5432/postgres"


@pytest.mark.asyncio
async def test_delete_thread_mirrors_supabase_cascade_delete(captured_execute):
    await dual_write.delete_thread("thread-1")

    assert len(captured_execute) == 1
    sql, args = captured_execute[0]
    assert "delete from public.threads where id = $1" in sql
    assert args == ("thread-1",)


@pytest.mark.asyncio
async def test_remove_thread_file_mirrors_association_rpc_only(captured_execute):
    await dual_write.remove_thread_file("thread-1", "file-1")

    assert len(captured_execute) == 1
    sql, args = captured_execute[0]
    assert "select public.remove_thread_file_pair($1, $2)" in sql
    assert "delete from public.files" not in sql
    assert args == ("thread-1", "file-1")


@pytest.mark.asyncio
async def test_delete_file_mirrors_file_row_delete(captured_execute):
    await dual_write.delete_file("file-1")

    assert len(captured_execute) == 1
    sql, args = captured_execute[0]
    assert "delete from public.files where file_hash = $1" in sql
    assert args == ("file-1",)


@pytest.mark.asyncio
async def test_delete_annotations_mirrors_pair_or_thread_scope(captured_execute):
    await dual_write.delete_annotations("thread-1", "file-1")
    await dual_write.delete_annotations("thread-1")

    assert len(captured_execute) == 2
    pair_sql, pair_args = captured_execute[0]
    thread_sql, thread_args = captured_execute[1]
    assert "where thread_id = $1 and file_hash = $2" in pair_sql
    assert pair_args == ("thread-1", "file-1")
    assert "where thread_id = $1" in thread_sql
    assert "file_hash = $2" not in thread_sql
    assert thread_args == ("thread-1",)


@pytest.mark.asyncio
async def test_delete_message_pair_mirrors_supabase_rpc(captured_execute):
    await dual_write.delete_message_pair("message-1")

    assert len(captured_execute) == 1
    sql, args = captured_execute[0]
    assert "select public.delete_message_pair($1)" in sql
    assert args == ("message-1",)
