"""Allow durable steering control commands.

Revision ID: a8d3e6f1b2c4
Revises: 2a6c8e1f4b9d
"""
from alembic import op

revision = "a8d3e6f1b2c4"
down_revision = "2a6c8e1f4b9d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("ck_agent_task_commands_action", "agent_task_commands", type_="check")
    op.create_check_constraint(
        "ck_agent_task_commands_action", "agent_task_commands",
        "action in ('start','pause','resume','cancel','retry','expire','delete','steer')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_agent_task_commands_action", "agent_task_commands", type_="check")
    op.create_check_constraint(
        "ck_agent_task_commands_action", "agent_task_commands",
        "action in ('start','pause','resume','cancel','retry','expire','delete')",
    )
