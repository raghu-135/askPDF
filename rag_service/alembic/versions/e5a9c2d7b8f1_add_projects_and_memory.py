"""Add projects and durable memory tables

Revision ID: e5a9c2d7b8f1
Revises: d4b9c7e2a1f0
Create Date: 2026-07-24 00:00:00.000000

"""
from __future__ import annotations

import uuid

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "e5a9c2d7b8f1"
down_revision = "d4b9c7e2a1f0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False, server_default=""),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("settings_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "embedding_model", name="uq_projects_id_embedding_model"),
    )
    op.create_index("ix_projects_name", "projects", ["name"], unique=False)
    op.create_index("idx_project_created_at", "projects", ["created_at"], unique=False)

    op.add_column("threads", sa.Column("project_id", sa.String(), nullable=True))
    op.create_index("ix_threads_project_id", "threads", ["project_id"], unique=False)

    default_project_id = str(uuid.uuid4())
    op.execute(
        sa.text(
            """
            insert into projects (id, name, description, embedding_model, settings_json, created_at)
            values (:id, 'Personal', 'Default project for existing threads.', 'BAAI/bge-m3', '{}'::jsonb, now())
            """
        ).bindparams(id=default_project_id)
    )
    op.execute(sa.text("update threads set project_id = :id where project_id is null").bindparams(id=default_project_id))
    op.execute("update threads set embedding_model = 'BAAI/bge-m3'")
    op.alter_column("threads", "project_id", nullable=False)
    op.create_foreign_key(
        "fk_threads_project_embedding_model",
        "threads",
        "projects",
        ["project_id", "embedding_model"],
        ["id", "embedding_model"],
        ondelete="RESTRICT",
    )

    op.create_table(
        "memories",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("scope_type", sa.String(), nullable=False, server_default="thread"),
        sa.Column("scope_id", sa.String(), nullable=False),
        sa.Column("memory_type", sa.String(), nullable=False, server_default="semantic"),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("summary", sa.String(), nullable=False, server_default=""),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("index_status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("index_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("index_error", sa.String(), nullable=True),
        sa.Column("source_refs_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
        sa.Column("visibility", sa.String(), nullable=False, server_default="private"),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fork_origin_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memories_scope_type", "memories", ["scope_type"], unique=False)
    op.create_index("ix_memories_scope_id", "memories", ["scope_id"], unique=False)
    op.create_index("ix_memories_memory_type", "memories", ["memory_type"], unique=False)
    op.create_index("ix_memories_embedding_model", "memories", ["embedding_model"], unique=False)
    op.create_index("ix_memories_content_hash", "memories", ["content_hash"], unique=False)
    op.create_index("ix_memories_index_status", "memories", ["index_status"], unique=False)
    op.create_index("ix_memories_status", "memories", ["status"], unique=False)
    op.create_index("ix_memories_visibility", "memories", ["visibility"], unique=False)
    op.create_index("ix_memories_created_by", "memories", ["created_by"], unique=False)
    op.create_index("idx_memory_scope_status", "memories", ["scope_type", "scope_id", "status"], unique=False)
    op.create_index("idx_memory_index_retry", "memories", ["index_status", "updated_at"], unique=False)
    op.create_index("idx_memory_created_at", "memories", ["created_at"], unique=False)

    op.create_table(
        "memory_events",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("memory_id", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("actor_id", sa.String(), nullable=True),
        sa.Column("payload_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["memory_id"], ["memories.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_events_memory_id", "memory_events", ["memory_id"], unique=False)
    op.create_index("ix_memory_events_event_type", "memory_events", ["event_type"], unique=False)
    op.create_index("ix_memory_events_actor_id", "memory_events", ["actor_id"], unique=False)

    op.create_table(
        "memory_candidates",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("source_thread_id", sa.String(), nullable=True),
        sa.Column("source_project_id", sa.String(), nullable=True),
        sa.Column("source_agent_run_id", sa.String(), nullable=True),
        sa.Column("source_turn_id", sa.String(), nullable=True),
        sa.Column("proposed_scope_type", sa.String(), nullable=False, server_default="thread"),
        sa.Column("proposed_scope_id", sa.String(), nullable=False),
        sa.Column("memory_type", sa.String(), nullable=False, server_default="semantic"),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("reason", sa.String(), nullable=False, server_default=""),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("promoted_memory_id", sa.String(), nullable=True),
        sa.Column("resolved_by", sa.String(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["promoted_memory_id"], ["memories.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    for column in ("source_thread_id", "source_project_id", "source_agent_run_id", "source_turn_id", "proposed_scope_type", "proposed_scope_id", "memory_type", "status", "promoted_memory_id", "resolved_by", "created_by"):
        op.create_index(f"ix_memory_candidates_{column}", "memory_candidates", [column], unique=False)
    op.create_index("idx_memory_candidate_scope_status", "memory_candidates", ["proposed_scope_type", "proposed_scope_id", "status"], unique=False)
    op.create_index("idx_memory_candidate_source_thread", "memory_candidates", ["source_thread_id", "created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_memory_candidate_source_thread", table_name="memory_candidates")
    op.drop_index("idx_memory_candidate_scope_status", table_name="memory_candidates")
    for column in ("created_by", "resolved_by", "promoted_memory_id", "status", "memory_type", "proposed_scope_id", "proposed_scope_type", "source_turn_id", "source_agent_run_id", "source_project_id", "source_thread_id"):
        op.drop_index(f"ix_memory_candidates_{column}", table_name="memory_candidates")
    op.drop_table("memory_candidates")

    op.drop_index("ix_memory_events_actor_id", table_name="memory_events")
    op.drop_index("ix_memory_events_event_type", table_name="memory_events")
    op.drop_index("ix_memory_events_memory_id", table_name="memory_events")
    op.drop_table("memory_events")

    op.drop_index("idx_memory_created_at", table_name="memories")
    op.drop_index("idx_memory_index_retry", table_name="memories")
    op.drop_index("idx_memory_scope_status", table_name="memories")
    op.drop_index("ix_memories_created_by", table_name="memories")
    op.drop_index("ix_memories_visibility", table_name="memories")
    op.drop_index("ix_memories_status", table_name="memories")
    op.drop_index("ix_memories_memory_type", table_name="memories")
    op.drop_index("ix_memories_index_status", table_name="memories")
    op.drop_index("ix_memories_content_hash", table_name="memories")
    op.drop_index("ix_memories_embedding_model", table_name="memories")
    op.drop_index("ix_memories_scope_id", table_name="memories")
    op.drop_index("ix_memories_scope_type", table_name="memories")
    op.drop_table("memories")

    op.drop_index("ix_threads_project_id", table_name="threads")
    op.drop_constraint("fk_threads_project_embedding_model", "threads", type_="foreignkey")
    op.drop_column("threads", "project_id")
    op.drop_index("idx_project_created_at", table_name="projects")
    op.drop_index("ix_projects_name", table_name="projects")
    op.drop_table("projects")
