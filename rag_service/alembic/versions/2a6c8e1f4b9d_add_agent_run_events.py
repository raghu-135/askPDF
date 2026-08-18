"""Add canonical agent run observability events.

Revision ID: 2a6c8e1f4b9d
Revises: 1e8f3a7c5b2d
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "2a6c8e1f4b9d"
down_revision = "1e8f3a7c5b2d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_run_events",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("agent_run_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("payload_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("trace_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint("attempt >= 1", name="ck_agent_run_events_attempt"),
        sa.CheckConstraint("sequence >= 0", name="ck_agent_run_events_sequence"),
        sa.ForeignKeyConstraint(["agent_run_id"], ["agent_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("agent_run_id", "event_id", name="uq_agent_run_events_run_event"),
    )
    op.create_index("ix_agent_run_events_agent_run_id", "agent_run_events", ["agent_run_id"])
    op.create_index("ix_agent_run_events_event_id", "agent_run_events", ["event_id"])
    op.create_index("ix_agent_run_events_kind", "agent_run_events", ["kind"])
    op.create_index("ix_agent_run_events_trace_id", "agent_run_events", ["trace_id"])
    op.create_index("idx_agent_run_events_run_sequence", "agent_run_events", ["agent_run_id", "attempt", "sequence"])


def downgrade() -> None:
    op.drop_table("agent_run_events")
