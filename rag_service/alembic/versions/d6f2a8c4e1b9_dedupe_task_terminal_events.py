"""Add causal idempotency for product task lifecycle events."""

from alembic import op
import sqlalchemy as sa


revision = "d6f2a8c4e1b9"
down_revision = "c3e9f5a7b2d4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_task_events", sa.Column("causal_key", sa.String(), nullable=True))
    op.execute(sa.text("""
        delete from agent_task_events duplicate
        using agent_task_events retained
        where duplicate.task_id = retained.task_id
          and duplicate.agent_run_id = retained.agent_run_id
          and duplicate.terminal is true
          and retained.terminal is true
          and (duplicate.sequence > retained.sequence
               or (duplicate.sequence = retained.sequence and duplicate.id > retained.id))
    """))
    op.execute(sa.text("""
        update agent_task_events
        set causal_key = concat('run', ':', agent_run_id, ':', 'terminal')
        where terminal is true and agent_run_id is not null
    """))
    op.create_index("ix_agent_task_events_causal_key", "agent_task_events", ["causal_key"])
    op.create_unique_constraint(
        "uq_agent_task_event_causal_key", "agent_task_events", ["task_id", "causal_key"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_agent_task_event_causal_key", "agent_task_events", type_="unique")
    op.drop_index("ix_agent_task_events_causal_key", table_name="agent_task_events")
    op.drop_column("agent_task_events", "causal_key")
