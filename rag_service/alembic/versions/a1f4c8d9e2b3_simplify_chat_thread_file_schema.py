"""Simplify chat, thread stats, and file annotation schema

Revision ID: a1f4c8d9e2b3
Revises: 2f6c9d1e8a4b
Create Date: 2026-06-28 00:00:00.000000

"""
from __future__ import annotations

import uuid

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "a1f4c8d9e2b3"
down_revision = "2f6c9d1e8a4b"
branch_labels = None
depends_on = None


def _table_exists(connection, table_name: str) -> bool:
    return bool(
        connection.execute(
            sa.text(
                """
                select exists (
                    select 1
                    from information_schema.tables
                    where table_schema = 'public'
                      and table_name = :table_name
                )
                """
            ),
            {"table_name": table_name},
        ).scalar()
    )


def _column_exists(connection, table_name: str, column_name: str) -> bool:
    return bool(
        connection.execute(
            sa.text(
                """
                select exists (
                    select 1
                    from information_schema.columns
                    where table_schema = 'public'
                      and table_name = :table_name
                      and column_name = :column_name
                )
                """
            ),
            {"table_name": table_name, "column_name": column_name},
        ).scalar()
    )


def _message_payload(user_row, assistant_row=None, *, error=None):
    question = user_row["content"] if user_row else ""
    rewritten_question = user_row["context_compact"] if user_row else None
    answer = assistant_row["content"] if assistant_row else None
    metadata = {}
    if assistant_row and assistant_row["context_compact"]:
        metadata["context_compact"] = assistant_row["context_compact"]

    return {
        "question": question,
        "rewritten_question": rewritten_question,
        "answer": answer,
        "reasoning": assistant_row["reasoning"] if assistant_row else "",
        "reasoning_available": bool(assistant_row["reasoning_available"]) if assistant_row else False,
        "reasoning_format": assistant_row["reasoning_format"] if assistant_row else "none",
        "web_sources": assistant_row["web_sources"] if assistant_row and assistant_row["web_sources"] else [],
        "document_sources": [],
        "used_chat_ids": [],
        "clarification_options": None,
        "error": error,
        "metadata": metadata,
    }


def _create_chat_turns() -> None:
    op.create_table(
        "chat_turns",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("thread_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="completed"),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["thread_id"], ["threads.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_chat_turn_thread_created", "chat_turns", ["thread_id", "created_at"], unique=False)
    op.create_index(op.f("ix_chat_turns_status"), "chat_turns", ["status"], unique=False)
    op.create_index(op.f("ix_chat_turns_thread_id"), "chat_turns", ["thread_id"], unique=False)


def _migrate_messages_to_chat_turns(connection) -> None:
    if not _table_exists(connection, "messages"):
        return

    rows = list(
        connection.execute(
            sa.text(
                """
                select id, thread_id, role, content, context_compact, reasoning,
                       reasoning_available, reasoning_format, web_sources, created_at
                from messages
                order by thread_id, created_at, id
                """
            )
        ).mappings()
    )

    grouped = {}
    for row in rows:
        grouped.setdefault(row["thread_id"], []).append(row)

    insert_rows = []
    for thread_id, messages in grouped.items():
        index = 0
        while index < len(messages):
            current = messages[index]
            if current["role"] == "user":
                assistant = None
                if index + 1 < len(messages) and messages[index + 1]["role"] == "assistant":
                    assistant = messages[index + 1]
                    index += 2
                else:
                    index += 1

                error = None
                status = "completed"
                completed_at = assistant["created_at"] if assistant else None
                if assistant is None:
                    status = "failed"
                    error = {
                        "code": "missing_assistant_message",
                        "raw_message": "Legacy user message had no immediately following assistant message.",
                        "retryable": False,
                    }

                insert_rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "thread_id": thread_id,
                        "status": status,
                        "payload": _message_payload(current, assistant, error=error),
                        "created_at": current["created_at"],
                        "completed_at": completed_at,
                    }
                )
                continue

            error = {
                "code": "missing_user_message",
                "raw_message": "Legacy assistant message had no immediately preceding user message.",
                "retryable": False,
            }
            insert_rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "thread_id": thread_id,
                    "status": "completed",
                    "payload": _message_payload(None, current, error=error),
                    "created_at": current["created_at"],
                    "completed_at": current["created_at"],
                }
            )
            index += 1

    if insert_rows:
        connection.execute(
            sa.text(
                """
                insert into chat_turns (id, thread_id, status, payload, created_at, completed_at)
                values (:id, :thread_id, :status, :payload, :created_at, :completed_at)
                """
            ).bindparams(sa.bindparam("payload", type_=postgresql.JSONB)),
            insert_rows,
        )

    op.drop_table("messages")


def upgrade() -> None:
    connection = op.get_bind()

    if not _table_exists(connection, "chat_turns"):
        _create_chat_turns()
    _migrate_messages_to_chat_turns(connection)

    if not _column_exists(connection, "threads", "total_qa_pairs"):
        op.add_column("threads", sa.Column("total_qa_pairs", sa.Integer(), server_default="0", nullable=False))
    if not _column_exists(connection, "threads", "total_qa_chars"):
        op.add_column("threads", sa.Column("total_qa_chars", sa.Integer(), server_default="0", nullable=False))
    if not _column_exists(connection, "threads", "avg_qa_chars"):
        op.add_column("threads", sa.Column("avg_qa_chars", sa.Float(), server_default="0", nullable=False))
    if not _column_exists(connection, "threads", "last_qa_at"):
        op.add_column("threads", sa.Column("last_qa_at", sa.DateTime(timezone=True), nullable=True))
    if not _column_exists(connection, "threads", "documents_meta"):
        op.add_column(
            "threads",
            sa.Column(
                "documents_meta",
                postgresql.JSONB(astext_type=sa.Text()),
                server_default=sa.text("'{}'::jsonb"),
                nullable=False,
            ),
        )
    if not _column_exists(connection, "threads", "stats_last_updated_at"):
        op.add_column(
            "threads",
            sa.Column(
                "stats_last_updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=True,
            ),
        )

    if _table_exists(connection, "thread_stats"):
        connection.execute(
            sa.text(
                """
                update threads t
                   set total_qa_pairs = coalesce(s.total_qa_pairs, 0),
                       total_qa_chars = coalesce(s.total_qa_chars, 0),
                       avg_qa_chars = coalesce(s.avg_qa_chars, 0),
                       last_qa_at = s.last_qa_at,
                       documents_meta = coalesce(s.documents_meta, '{}'::jsonb),
                       stats_last_updated_at = coalesce(s.last_updated_at, now())
                  from thread_stats s
                 where s.thread_id = t.id
                """
            )
        )
        op.drop_table("thread_stats")

    if not _column_exists(connection, "thread_files", "annotations"):
        op.add_column(
            "thread_files",
            sa.Column(
                "annotations",
                postgresql.JSONB(astext_type=sa.Text()),
                server_default=sa.text("'[]'::jsonb"),
                nullable=False,
            ),
        )
    if not _column_exists(connection, "thread_files", "annotations_updated_at"):
        op.add_column(
            "thread_files",
            sa.Column("annotations_updated_at", sa.DateTime(timezone=True), nullable=True),
        )

    if _table_exists(connection, "thread_file_annotations"):
        connection.execute(
            sa.text(
                """
                update thread_files tf
                   set annotations = coalesce(tfa.annotations_json::jsonb, '[]'::jsonb),
                       annotations_updated_at = case
                           when coalesce(tfa.annotations_json::jsonb, '[]'::jsonb) = '[]'::jsonb
                           then null
                           else tfa.updated_at
                       end
                  from thread_file_annotations tfa
                 where tfa.thread_id = tf.thread_id
                   and tfa.file_hash = tf.file_hash
                """
            )
        )
        op.drop_table("thread_file_annotations")

    if _column_exists(connection, "threads", "settings"):
        connection.execute(sa.text("update threads set settings = settings - 'updated_at' where settings ? 'updated_at'"))
    if _column_exists(connection, "threads", "thread_metadata"):
        connection.execute(sa.text("update threads set thread_metadata = thread_metadata - 'updated_at' where thread_metadata ? 'updated_at'"))
    if _column_exists(connection, "threads", "documents_meta"):
        connection.execute(sa.text("update threads set documents_meta = documents_meta - 'updated_at' where documents_meta ? 'updated_at'"))
    if _table_exists(connection, "chat_turns"):
        connection.execute(sa.text("update chat_turns set payload = payload - 'updated_at' where payload ? 'updated_at'"))


def downgrade() -> None:
    connection = op.get_bind()

    if _table_exists(connection, "chat_turns") and not _table_exists(connection, "messages"):
        op.create_table(
            "messages",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("thread_id", sa.String(), nullable=True),
            sa.Column("role", sa.String(), nullable=False),
            sa.Column("content", sa.String(), nullable=False),
            sa.Column("context_compact", sa.String(), nullable=True),
            sa.Column("reasoning", sa.String(), nullable=True),
            sa.Column("reasoning_available", sa.Boolean(), nullable=False),
            sa.Column("reasoning_format", sa.String(), nullable=False),
            sa.Column("web_sources", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
            sa.ForeignKeyConstraint(["thread_id"], ["threads.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_messages_thread_id"), "messages", ["thread_id"], unique=False)
        op.create_index(op.f("ix_messages_role"), "messages", ["role"], unique=False)

        connection.execute(
            sa.text(
                """
                insert into messages (
                    id, thread_id, role, content, context_compact, reasoning,
                    reasoning_available, reasoning_format, web_sources, created_at
                )
                select id || ':user',
                       thread_id,
                       'user',
                       coalesce(payload->>'question', ''),
                       payload->>'rewritten_question',
                       null,
                       false,
                       'none',
                       '[]'::jsonb,
                       created_at
                  from chat_turns
                 where payload ? 'question'
                union all
                select id || ':assistant',
                       thread_id,
                       'assistant',
                       coalesce(payload->>'answer', ''),
                       payload->'metadata'->>'context_compact',
                       payload->>'reasoning',
                       coalesce((payload->>'reasoning_available')::boolean, false),
                       coalesce(payload->>'reasoning_format', 'none'),
                       coalesce(payload->'web_sources', '[]'::jsonb),
                       coalesce(completed_at, created_at)
                  from chat_turns
                 where payload->>'answer' is not null
                """
            )
        )

    if _table_exists(connection, "thread_files") and not _table_exists(connection, "thread_file_annotations"):
        op.create_table(
            "thread_file_annotations",
            sa.Column("thread_id", sa.String(), nullable=False),
            sa.Column("file_hash", sa.String(), nullable=False),
            sa.Column("annotations_json", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["file_hash"], ["files.file_hash"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["thread_id"], ["threads.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("thread_id", "file_hash"),
        )

    if _column_exists(connection, "thread_files", "annotations"):
        connection.execute(
            sa.text(
                """
                insert into thread_file_annotations (
                    thread_id,
                    file_hash,
                    annotations_json,
                    created_at,
                    updated_at
                )
                select thread_id,
                       file_hash,
                       coalesce(annotations, '[]'::jsonb)::text,
                       added_at,
                       annotations_updated_at
                  from thread_files
                 where coalesce(annotations, '[]'::jsonb) <> '[]'::jsonb
                on conflict (thread_id, file_hash) do update
                    set annotations_json = excluded.annotations_json,
                        updated_at = excluded.updated_at
                """
            )
        )

    if not _table_exists(connection, "thread_stats"):
        op.create_table(
            "thread_stats",
            sa.Column("thread_id", sa.String(), nullable=False),
            sa.Column("total_qa_pairs", sa.Integer(), nullable=False),
            sa.Column("total_qa_chars", sa.Integer(), nullable=False),
            sa.Column("avg_qa_chars", sa.Float(), nullable=False),
            sa.Column("last_qa_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("documents_meta", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("last_updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
            sa.ForeignKeyConstraint(["thread_id"], ["threads.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("thread_id"),
        )

    if _column_exists(connection, "threads", "total_qa_pairs"):
        connection.execute(
            sa.text(
                """
                insert into thread_stats (
                    thread_id,
                    total_qa_pairs,
                    total_qa_chars,
                    avg_qa_chars,
                    last_qa_at,
                    documents_meta,
                    last_updated_at
                )
                select id,
                       coalesce(total_qa_pairs, 0),
                       coalesce(total_qa_chars, 0),
                       coalesce(avg_qa_chars, 0),
                       last_qa_at,
                       coalesce(documents_meta, '{}'::jsonb),
                       coalesce(stats_last_updated_at, now())
                  from threads
                on conflict (thread_id) do update
                    set total_qa_pairs = excluded.total_qa_pairs,
                        total_qa_chars = excluded.total_qa_chars,
                        avg_qa_chars = excluded.avg_qa_chars,
                        last_qa_at = excluded.last_qa_at,
                        documents_meta = excluded.documents_meta,
                        last_updated_at = excluded.last_updated_at
                """
            )
        )

    if _column_exists(connection, "thread_files", "annotations_updated_at"):
        op.drop_column("thread_files", "annotations_updated_at")
    if _column_exists(connection, "thread_files", "annotations"):
        op.drop_column("thread_files", "annotations")

    for column_name in (
        "stats_last_updated_at",
        "documents_meta",
        "last_qa_at",
        "avg_qa_chars",
        "total_qa_chars",
        "total_qa_pairs",
    ):
        if _column_exists(connection, "threads", column_name):
            op.drop_column("threads", column_name)

    if _table_exists(connection, "chat_turns"):
        op.drop_index(op.f("ix_chat_turns_thread_id"), table_name="chat_turns")
        op.drop_index(op.f("ix_chat_turns_status"), table_name="chat_turns")
        op.drop_index("idx_chat_turn_thread_created", table_name="chat_turns")
        op.drop_table("chat_turns")
