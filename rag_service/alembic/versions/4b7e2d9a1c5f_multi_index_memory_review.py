"""Add Global memory representations and consistency review state.

Revision ID: 4b7e2d9a1c5f
Revises: 3a8d7c5e1f2b
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "4b7e2d9a1c5f"
down_revision = "3a8d7c5e1f2b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "global_memory_representations",
        sa.Column("memory_id", sa.String(), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("index_status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("index_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("index_error", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("length(btrim(embedding_model)) > 0", name="ck_global_memory_rep_model_nonempty"),
        sa.CheckConstraint("length(btrim(content_hash)) > 0", name="ck_global_memory_rep_hash_nonempty"),
        sa.CheckConstraint("index_status in ('pending', 'indexing', 'indexed', 'failed')", name="ck_global_memory_rep_status"),
        sa.CheckConstraint("index_attempts >= 0", name="ck_global_memory_rep_attempts"),
        sa.ForeignKeyConstraint(["memory_id"], ["memories.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("memory_id", "embedding_model"),
    )
    op.create_index(
        "idx_global_memory_rep_retry",
        "global_memory_representations",
        ["embedding_model", "index_status", "updated_at"],
    )
    op.create_table(
        "memory_scope_activity",
        sa.Column("scope_type", sa.String(), nullable=False),
        sa.Column("scope_id", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("changed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("scope_type in ('user', 'project', 'thread')", name="ck_memory_scope_activity_type"),
        sa.CheckConstraint("version >= 1", name="ck_memory_scope_activity_version"),
        sa.PrimaryKeyConstraint("scope_type", "scope_id"),
    )
    op.create_table(
        "memory_review_states",
        sa.Column("context_type", sa.String(), nullable=False),
        sa.Column("context_id", sa.String(), nullable=False),
        sa.Column("reviewed_scope_versions_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("last_reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("context_type in ('project', 'thread')", name="ck_memory_review_context_type"),
        sa.PrimaryKeyConstraint("context_type", "context_id"),
    )
    op.execute(
        """
        insert into global_memory_representations
            (memory_id, embedding_model, content_hash, index_status, index_attempts)
        select distinct m.id, p.embedding_model, m.content_hash, 'pending', 0
          from memories m
          cross join projects p
         where m.scope_type = 'user'
           and m.scope_id = 'default'
           and p.embedding_model <> m.embedding_model
        on conflict do nothing
        """
    )
    op.execute(
        """
        insert into memory_scope_activity (scope_type, scope_id, version, changed_at)
        select scope_type, scope_id, 1, coalesce(max(updated_at), max(created_at), now())
          from memories
         group by scope_type, scope_id
        on conflict do nothing
        """
    )


def downgrade() -> None:
    op.drop_table("memory_review_states")
    op.drop_table("memory_scope_activity")
    op.drop_index("idx_global_memory_rep_retry", table_name="global_memory_representations")
    op.drop_table("global_memory_representations")
