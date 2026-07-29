"""
test_schema_guardrails.py - Tests that protect the simplified Postgres schema.
"""


from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
import pytest
from sqlalchemy import text


from app.db import models_sqlmodel


EXPECTED_MODEL_TABLES = {
    "projects",
    "chat_turns",
    "files",
    "memories",
    "memory_candidates",
    "memory_events",
    "project_files",
    "thread_files",
    "threads",
}

REMOVED_TABLES = {
    "messages",
    "messages_legacy",
    "thread_stats",
    "thread_file_annotations",
}

REMOVED_MODEL_EXPORTS = {
    "Message",
    "ThreadStats",
    "ThreadFileAnnotation",
}


def test_alembic_graph_retains_applied_memory_compatibility_revisions():
    service_root = Path(__file__).resolve().parents[1]
    config = Config(str(service_root / "alembic.ini"))
    config.set_main_option("script_location", str(service_root / "alembic"))
    scripts = ScriptDirectory.from_config(config)

    cleanup = scripts.get_revision("c9e6a1b4d3f8")

    assert scripts.get_heads() == ["e2c7a9f4b1d6"]
    assert scripts.get_revision("a7c4e9f2b1d6") is not None
    assert scripts.get_revision("b8d5f0a3c2e7") is not None
    assert cleanup is not None
    assert cleanup.down_revision == "b8d5f0a3c2e7"
    assert scripts.get_revision("e2c7a9f4b1d6").down_revision == "c9e6a1b4d3f8"


def test_project_files_has_composite_key_and_cascading_foreign_keys():
    table = models_sqlmodel.ProjectFile.__table__
    assert [column.name for column in table.primary_key.columns] == ["project_id", "file_hash"]
    targets = {
        element.target_fullname: element.parent.name
        for constraint in table.foreign_key_constraints
        for element in constraint.elements
        if constraint.ondelete == "CASCADE"
    }
    assert targets == {
        "projects.id": "project_id",
        "files.file_hash": "file_hash",
    }


def test_sqlmodel_metadata_only_contains_current_application_tables():
    table_names = set(models_sqlmodel.SQLModel.metadata.tables.keys())

    assert EXPECTED_MODEL_TABLES.issubset(table_names)
    assert table_names.isdisjoint(REMOVED_TABLES)


def test_removed_orm_models_are_not_exported():
    for model_name in REMOVED_MODEL_EXPORTS:
        assert not hasattr(models_sqlmodel, model_name)


def test_thread_model_has_one_strict_project_embedding_foreign_key():
    assert not hasattr(models_sqlmodel.Thread, "is_legacy")
    foreign_keys = {
        constraint.name: constraint
        for constraint in models_sqlmodel.Thread.__table__.constraints
        if constraint.name and constraint.__class__.__name__ == "ForeignKeyConstraint"
    }

    constraint = foreign_keys["fk_threads_project_embedding_model"]
    assert [column.name for column in constraint.columns] == [
        "project_id",
        "embedding_model",
    ]
    assert [element.target_fullname for element in constraint.elements] == [
        "projects.id",
        "projects.embedding_model",
    ]


def test_memory_models_define_hardening_check_constraints():
    memory_constraints = {
        constraint.name
        for constraint in models_sqlmodel.Memory.__table__.constraints
        if constraint.name
    }
    candidate_constraints = {
        constraint.name
        for constraint in models_sqlmodel.MemoryCandidate.__table__.constraints
        if constraint.name
    }

    assert {
        "ck_memories_scope_type",
        "ck_memories_memory_type",
        "ck_memories_status",
        "ck_memories_visibility",
        "ck_memories_confidence",
        "ck_memories_scope_id_nonempty",
        "ck_memories_content_nonempty",
    }.issubset(memory_constraints)
    assert {
        "ck_memory_candidates_scope_type",
        "ck_memory_candidates_memory_type",
        "ck_memory_candidates_status",
        "ck_memory_candidates_confidence",
        "ck_memory_candidates_scope_id_nonempty",
        "ck_memory_candidates_content_nonempty",
    }.issubset(candidate_constraints)


@pytest.mark.asyncio
async def test_created_database_schema_excludes_removed_tables(engine):
    async with engine.connect() as connection:
        result = await connection.execute(
            text(
                """
                select table_name
                  from information_schema.tables
                 where table_schema = 'public'
                   and table_type = 'BASE TABLE'
                """
            )
        )
        table_names = {row[0] for row in result}

    assert EXPECTED_MODEL_TABLES.issubset(table_names)
    assert table_names.isdisjoint(REMOVED_TABLES)
