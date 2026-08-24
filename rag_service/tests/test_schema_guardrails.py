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
    "memory_events",
    "memory_overrides",
    "global_memory_representations",
    "memory_scope_activity",
    "memory_review_states",
    "memory_manager_idempotency",
    "embedding_jobs",
    "project_files",
    "thread_document_annotations",
    "thread_files",
    "threads",
}

REMOVED_TABLES = {
    "messages",
    "messages_legacy",
    "thread_stats",
    "thread_file_annotations",
    "memory_candidates",
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

    assert set(scripts.get_heads()) == {"k5e2a8c4d7f1"}
    assert scripts.get_revision("k5e2a8c4d7f1").down_revision == "j4d9e6f1b3c5"
    assert scripts.get_revision("9b4d6e2f1a7c").down_revision == "a8d3f1c6e4b2"
    assert scripts.get_revision("a8d3f1c6e4b2").down_revision == "e7c4a1b9d2f6"
    assert scripts.get_revision("e7c4a1b9d2f6").down_revision == "d5f1a2b3c4e6"
    assert scripts.get_revision("a7c4e9f2b1d6") is not None
    assert scripts.get_revision("b8d5f0a3c2e7") is not None
    assert cleanup is not None
    assert cleanup.down_revision == "b8d5f0a3c2e7"
    assert scripts.get_revision("e2c7a9f4b1d6").down_revision == "c9e6a1b4d3f8"
    assert scripts.get_revision("f4b8c2d7e1a9").down_revision == "e2c7a9f4b1d6"
    assert scripts.get_revision("d9a4e7c2b1f6").down_revision == "f4b8c2d7e1a9"
    assert scripts.get_revision("1c7d9e4a2b6f").down_revision == "d9a4e7c2b1f6"
    assert scripts.get_revision("3a8d7c5e1f2b").down_revision == "1c7d9e4a2b6f"
    assert scripts.get_revision("4b7e2d9a1c5f").down_revision == "3a8d7c5e1f2b"
    assert scripts.get_revision("6d2f8a9b3c1e").down_revision == "4b7e2d9a1c5f"
    assert scripts.get_revision("d1e8f4a9b2c7").down_revision == "6d2f8a9b3c1e"
    assert scripts.get_revision("7f3c1a9d5e2b").down_revision == "d1e8f4a9b2c7"
    assert scripts.get_revision("c4e8a1b6d2f0").down_revision == "7f3c1a9d5e2b"


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


def test_projects_have_indexed_last_activity_timestamp():
    table = models_sqlmodel.Project.__table__

    assert table.c.last_activity_at.nullable is False
    assert "idx_project_last_activity_at" in {
        index.name for index in table.indexes
    }


def test_thread_document_annotations_are_thread_owned_overlays():
    table = models_sqlmodel.ThreadDocumentAnnotation.__table__
    assert [column.name for column in table.primary_key.columns] == ["thread_id", "file_hash"]
    targets = {
        element.target_fullname: element.parent.name
        for constraint in table.foreign_key_constraints
        for element in constraint.elements
        if constraint.ondelete == "CASCADE"
    }
    assert targets == {
        "threads.id": "thread_id",
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
    assert {
        "ck_memories_scope_type",
        "ck_memories_scope_id_nonempty",
        "ck_memories_content_nonempty",
    }.issubset(memory_constraints)


def test_memory_overrides_define_directional_key_and_cascades():
    table = models_sqlmodel.MemoryOverride.__table__
    assert [column.name for column in table.primary_key.columns] == [
        "overriding_memory_id",
        "overridden_memory_id",
    ]
    assert "ck_memory_overrides_not_self" in {
        constraint.name for constraint in table.constraints if constraint.name
    }
    assert "ix_memory_overrides_target" in {index.name for index in table.indexes}
    assert {
        element.parent.name
        for constraint in table.foreign_key_constraints
        for element in constraint.elements
        if constraint.ondelete == "CASCADE"
    } == {"overriding_memory_id", "overridden_memory_id"}


def test_global_memory_representations_are_model_aware_and_cascade():
    table = models_sqlmodel.GlobalMemoryRepresentation.__table__
    assert [column.name for column in table.primary_key.columns] == ["memory_id", "embedding_model"]
    assert "idx_global_memory_rep_retry" in {index.name for index in table.indexes}
    assert {
        element.parent.name
        for constraint in table.foreign_key_constraints
        for element in constraint.elements
        if constraint.ondelete == "CASCADE"
    } == {"memory_id"}


def test_embedding_jobs_are_durable_and_deduplicated():
    table = models_sqlmodel.EmbeddingJob.__table__
    assert [column.name for column in table.primary_key.columns] == ["id"]
    assert "uq_embedding_job_target" in {
        constraint.name for constraint in table.constraints if constraint.name
    }
    assert {"idx_embedding_job_claim", "idx_embedding_job_model_status"} <= {
        index.name for index in table.indexes
    }


def test_memory_review_state_tables_have_context_keys():
    assert [column.name for column in models_sqlmodel.MemoryScopeActivity.__table__.primary_key.columns] == [
        "scope_type", "scope_id"
    ]
    assert [column.name for column in models_sqlmodel.MemoryReviewState.__table__.primary_key.columns] == [
        "context_type", "context_id"
    ]
    constraints = {
        constraint.name: str(constraint.sqltext)
        for constraint in models_sqlmodel.MemoryReviewState.__table__.constraints
        if constraint.name
    }
    assert "user" in constraints["ck_memory_review_context_type"]


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
