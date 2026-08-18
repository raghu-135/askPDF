"""Rename thread embed_model to embedding_model

Revision ID: d4b9c7e2a1f0
Revises: f3a9b1c2d4e6
Create Date: 2026-07-17 00:00:00.000000

"""
from alembic import op


revision = "d4b9c7e2a1f0"
down_revision = "f3a9b1c2d4e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_threads_embed_model")
    op.alter_column("threads", "embed_model", new_column_name="embedding_model")
    op.create_index("ix_threads_embedding_model", "threads", ["embedding_model"], unique=False)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_threads_embedding_model")
    op.alter_column("threads", "embedding_model", new_column_name="embed_model")
    op.create_index("ix_threads_embed_model", "threads", ["embed_model"], unique=False)
