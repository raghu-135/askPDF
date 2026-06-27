import pytest

from app.db import supabase_client
from app.services.supabase_health_service import get_supabase_health


@pytest.mark.asyncio
async def test_supabase_health_disabled_without_flags(monkeypatch):
    for key in (
        "RAG_SUPABASE_DUAL_WRITE",
        "RAG_SUPABASE_PARITY_CHECK",
        "NEXT_PUBLIC_USE_SUPABASE_THREADS",
        "NEXT_PUBLIC_USE_SUPABASE_MESSAGES",
        "NEXT_PUBLIC_USE_SUPABASE_FILES",
        "NEXT_PUBLIC_USE_SUPABASE_REALTIME",
        "NEXT_PUBLIC_USE_SUPABASE_STORAGE",
        "RAG_SUPABASE_DB_URL",
        "SUPABASE_DB_URL",
        "TARGET_DATABASE_URL",
        "MIGRATION_TARGET_DATABASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    health = await get_supabase_health()

    assert health["status"] == "disabled"
    assert health["healthy"] is False
    assert health["enabled"] is False
    assert health["db"]["reachable"] is False


@pytest.mark.asyncio
async def test_supabase_health_reports_db_and_storage_ok(monkeypatch):
    monkeypatch.setenv("RAG_SUPABASE_DUAL_WRITE", "true")
    monkeypatch.setenv("NEXT_PUBLIC_USE_SUPABASE_STORAGE", "true")
    monkeypatch.setenv("RAG_SUPABASE_DB_URL", "postgresql://postgres:postgres@supabase-db:5432/postgres")

    async def fake_fetchval(sql, *args):
        if "public.threads" in sql:
            return 12
        if "storage.buckets" in sql:
            return True
        if "storage.objects" in sql:
            return 7
        return None

    monkeypatch.setattr(supabase_client, "fetchval", fake_fetchval)

    health = await get_supabase_health()

    assert health["status"] == "healthy"
    assert health["healthy"] is True
    assert health["db"]["thread_count"] == 12
    assert health["storage"]["bucket_exists"] is True
    assert health["storage"]["object_count"] == 7


@pytest.mark.asyncio
async def test_supabase_health_reports_degraded_db_error(monkeypatch):
    monkeypatch.setenv("RAG_SUPABASE_DUAL_WRITE", "true")
    monkeypatch.setenv("RAG_SUPABASE_DB_URL", "postgresql://postgres:postgres@supabase-db:5432/postgres")

    async def fake_fetchval(sql, *args):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(supabase_client, "fetchval", fake_fetchval)

    health = await get_supabase_health()

    assert health["status"] == "degraded"
    assert health["healthy"] is False
    assert health["db"]["reachable"] is False
    assert "connection refused" in health["db"]["error"]
