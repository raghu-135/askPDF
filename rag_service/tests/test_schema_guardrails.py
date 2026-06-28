"""
test_schema_guardrails.py - Tests that protect the simplified Postgres schema.
"""

import os
import sys

import pytest
from sqlalchemy import text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.db import models_sqlmodel


EXPECTED_MODEL_TABLES = {
    "chat_turns",
    "files",
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
