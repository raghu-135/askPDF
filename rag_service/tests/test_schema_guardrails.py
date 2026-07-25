"""
test_schema_guardrails.py - Tests that protect the simplified Postgres schema.
"""


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


def test_sqlmodel_metadata_only_contains_current_application_tables():
    table_names = set(models_sqlmodel.SQLModel.metadata.tables.keys())

    assert EXPECTED_MODEL_TABLES.issubset(table_names)
    assert table_names.isdisjoint(REMOVED_TABLES)


def test_removed_orm_models_are_not_exported():
    for model_name in REMOVED_MODEL_EXPORTS:
        assert not hasattr(models_sqlmodel, model_name)


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
