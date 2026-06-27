import pytest

from app.tools import db_migration_cli as cli


def test_migration_cli_requires_exactly_one_mode():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--dry-run", "--migrate"])
    args = parser.parse_args(["--validate-only"])
    assert args.validate_only is True


def test_redact_url_hides_password():
    redacted = cli.redact_url("postgresql://postgres:secret@localhost:5432/askpdf")
    assert "secret" not in redacted
    assert "postgres:***@localhost" in redacted


def test_checksum_rows_is_order_independent():
    left = [{"id": "b", "value": 2}, {"id": "a", "value": 1}]
    right = [{"id": "a", "value": 1}, {"id": "b", "value": 2}]
    assert cli.checksum_rows(left) == cli.checksum_rows(right)


def test_file_status_updated_at_is_ignored_for_comparison():
    left = {
        "file_hash": "abc",
        "file_status": {"updated_at": "2026-01-01T00:00:00.000001Z", "parsing": {"status": "completed"}},
    }
    right = {
        "file_hash": "abc",
        "file_status": {"updated_at": "2026-01-01T00:00:00.000002Z", "parsing": {"status": "completed"}},
    }
    assert cli.comparable_row("files", left) == cli.comparable_row("files", right)


def test_build_upsert_sql_uses_primary_key_conflict():
    sql = cli.build_upsert_sql("thread_files")
    assert "on conflict (thread_id, file_hash)" in sql
    assert "insert into public.thread_files" in sql


@pytest.mark.asyncio
async def test_apply_schema_if_needed_executes_sorted_sql_files(tmp_path):
    (tmp_path / "0002_second.sql").write_text("select 2;")
    (tmp_path / "0001_first.sql").write_text("select 1;")
    executed = []

    class FakeConn:
        async def fetchval(self, *args):
            return False

        async def execute(self, sql):
            executed.append(sql)

    await cli.apply_schema_if_needed(FakeConn(), str(tmp_path))
    assert executed == ["select 1;", "select 2;"]


@pytest.mark.asyncio
async def test_validate_storage_objects_reports_clean_parity(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXT_PUBLIC_USE_SUPABASE_STORAGE", "true")
    file_hash = "abc123"
    (tmp_path / f"{file_hash}.pdf").write_bytes(b"%PDF")

    class SourceConn:
        async def fetch(self, *args):
            return [{"file_hash": file_hash}]

    class TargetConn:
        async def fetchrow(self, *args):
            return {"storage_bucket": "pdfs", "storage_path": f"{file_hash}.pdf"}

        async def fetchval(self, *args):
            return True

    args = type("Args", (), {"static_dir": str(tmp_path), "storage_bucket": "pdfs"})()
    summary = await cli.validate_storage_objects(SourceConn(), TargetConn(), args)

    assert summary == {
        "enabled": True,
        "checked": 1,
        "missing_local": 0,
        "missing_metadata": [],
        "missing_objects": [],
        "missing_metadata_count": 0,
        "missing_object_count": 0,
    }


@pytest.mark.asyncio
async def test_validate_storage_objects_reports_missing_object(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXT_PUBLIC_USE_SUPABASE_STORAGE", "true")
    file_hash = "abc123"
    (tmp_path / f"{file_hash}.pdf").write_bytes(b"%PDF")

    class SourceConn:
        async def fetch(self, *args):
            return [{"file_hash": file_hash}]

    class TargetConn:
        async def fetchrow(self, *args):
            return {"storage_bucket": "pdfs", "storage_path": f"{file_hash}.pdf"}

        async def fetchval(self, *args):
            return False

    args = type("Args", (), {"static_dir": str(tmp_path), "storage_bucket": "pdfs"})()
    summary = await cli.validate_storage_objects(SourceConn(), TargetConn(), args)

    assert summary["missing_object_count"] == 1
    assert summary["missing_objects"] == [file_hash]
