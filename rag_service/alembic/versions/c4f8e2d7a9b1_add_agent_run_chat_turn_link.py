"""Add explicit chat turn link to agent runs.

Revision ID: c4f8e2d7a9b1
Revises: b7e2a4c9d1f0
Create Date: 2026-07-03
"""

from alembic import op
import sqlalchemy as sa


revision = "c4f8e2d7a9b1"
down_revision = "b7e2a4c9d1f0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_runs", sa.Column("chat_turn_id", sa.String(), nullable=True))
    op.create_foreign_key(
        "fk_agent_runs_chat_turn_id_chat_turns",
        "agent_runs",
        "chat_turns",
        ["chat_turn_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(op.f("ix_agent_runs_chat_turn_id"), "agent_runs", ["chat_turn_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_agent_runs_chat_turn_id"), table_name="agent_runs")
    op.drop_constraint("fk_agent_runs_chat_turn_id_chat_turns", "agent_runs", type_="foreignkey")
    op.drop_column("agent_runs", "chat_turn_id")
