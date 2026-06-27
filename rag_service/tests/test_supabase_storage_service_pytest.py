import pytest

from app.services import supabase_storage_service as storage_service


@pytest.mark.asyncio
async def test_storage_mirror_skips_when_flag_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXT_PUBLIC_USE_SUPABASE_STORAGE", "false")

    called = False

    async def fake_execute(*args):
        nonlocal called
        called = True

    monkeypatch.setattr(storage_service.supabase_client, "execute", fake_execute)

    pdf = tmp_path / "abc.pdf"
    pdf.write_bytes(b"%PDF")
    await storage_service.mirror_pdf_to_storage("abc", pdf)

    assert called is False


@pytest.mark.asyncio
async def test_storage_mirror_uploads_and_records_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXT_PUBLIC_USE_SUPABASE_STORAGE", "true")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("SUPABASE_STORAGE_URL", "http://storage.test/storage/v1")
    monkeypatch.setenv("SUPABASE_STORAGE_BUCKET", "pdfs")

    pdf = tmp_path / "abc.pdf"
    pdf.write_bytes(b"%PDF")
    posts = []
    writes = []

    class FakeResponse:
        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, headers, content):
            posts.append({"url": url, "headers": headers, "content": content})
            return FakeResponse()

    async def fake_execute(sql, *args):
        writes.append(args)

    monkeypatch.setattr(storage_service.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(storage_service.supabase_client, "execute", fake_execute)

    await storage_service.mirror_pdf_to_storage("abc", pdf)

    assert posts[0]["url"] == "http://storage.test/storage/v1/object/pdfs/abc.pdf"
    assert posts[0]["headers"]["authorization"] == "Bearer service-key"
    assert posts[0]["headers"]["x-upsert"] == "true"
    assert posts[0]["content"] == b"%PDF"
    assert writes == [("abc", "pdfs", "abc.pdf")]


@pytest.mark.asyncio
async def test_storage_mirror_failure_is_swallowed(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("NEXT_PUBLIC_USE_SUPABASE_STORAGE", "true")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")

    pdf = tmp_path / "abc.pdf"
    pdf.write_bytes(b"%PDF")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            raise RuntimeError("storage down")

    monkeypatch.setattr(storage_service.httpx, "AsyncClient", FakeClient)

    await storage_service.mirror_pdf_to_storage("abc", pdf)

    assert "storage down" in caplog.text
