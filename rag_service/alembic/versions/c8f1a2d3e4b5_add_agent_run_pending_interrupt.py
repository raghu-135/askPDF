"""Add pending interrupt storage to agent runs

Revision ID: c8f1a2d3e4b5
Revises: b7e2a4c9d1f0
Create Date: 2026-07-05 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "c8f1a2d3e4b5"
down_revision = "b7e2a4c9d1f0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column("pending_interrupt_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_runs", "pending_interrupt_json")
