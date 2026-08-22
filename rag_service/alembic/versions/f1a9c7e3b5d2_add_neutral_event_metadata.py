"""Add metadata required by newly written neutral runtime events."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "f1a9c7e3b5d2"
down_revision = "a8d3e6f1b2c4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_run_events", sa.Column("terminal", sa.Boolean(), nullable=False, server_default=sa.text("false")))
    op.add_column(
        "agent_run_events",
        sa.Column(
            "source_metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.add_column("agent_task_events", sa.Column("event_id", sa.String(), nullable=True))
    op.add_column("agent_task_events", sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("agent_task_events", sa.Column("terminal", sa.Boolean(), nullable=False, server_default=sa.text("false")))
    op.add_column(
        "agent_task_events",
        sa.Column(
            "source_metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.create_index("ix_agent_task_events_event_id", "agent_task_events", ["event_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_agent_task_events_event_id", table_name="agent_task_events")
    op.drop_column("agent_task_events", "source_metadata_json")
    op.drop_column("agent_task_events", "terminal")
    op.drop_column("agent_task_events", "occurred_at")
    op.drop_column("agent_task_events", "event_id")
    op.drop_column("agent_run_events", "source_metadata_json")
    op.drop_column("agent_run_events", "terminal")
