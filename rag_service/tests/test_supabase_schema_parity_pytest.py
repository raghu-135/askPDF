from pathlib import Path

from app.db.models_sqlmodel import (
    File,
    Message,
    Thread,
    ThreadFile,
    ThreadFileAnnotation,
    ThreadStats,
)


MIGRATION_SQL = Path("/supabase/migrations/0001_initial_schema.sql")
if not MIGRATION_SQL.exists():
    MIGRATION_SQL = Path(__file__).resolve().parents[2] / "supabase/migrations/0001_initial_schema.sql"


def _table_columns(model):
    return {column.name for column in model.__table__.columns}


def test_supabase_initial_schema_preserves_sqlmodel_columns():
    sql = MIGRATION_SQL.read_text()
    expected = {
        "threads": _table_columns(Thread),
        "files": _table_columns(File),
        "thread_files": _table_columns(ThreadFile),
        "messages": _table_columns(Message),
        "thread_file_annotations": _table_columns(ThreadFileAnnotation),
        "thread_stats": _table_columns(ThreadStats),
    }

    for table, columns in expected.items():
        assert f"create table if not exists public.{table}" in sql
        for column in columns:
            assert column in sql, f"{table}.{column} is missing from Supabase schema"


def test_supabase_initial_schema_enables_rls_and_realtime_for_core_tables():
    sql = MIGRATION_SQL.read_text()
    for table in (
        "threads",
        "files",
        "thread_files",
        "messages",
        "thread_file_annotations",
        "thread_stats",
    ):
        assert f"alter table public.{table} enable row level security" in sql
        assert f"alter publication supabase_realtime add table public.{table}" in sql
