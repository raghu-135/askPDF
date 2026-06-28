"""Add thread metadata

Revision ID: 2f6c9d1e8a4b
Revises: 8c8d6eac150a
Create Date: 2026-06-28 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "2f6c9d1e8a4b"
down_revision = "8c8d6eac150a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "threads",
        sa.Column(
            "thread_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("threads", "thread_metadata")
