"""Make agent workflow names unique

Revision ID: f3a9b1c2d4e6
Revises: e9a1b2c3d4f5
Create Date: 2026-07-08 00:00:00.000000

"""
from alembic import op


revision = "f3a9b1c2d4e6"
down_revision = "e9a1b2c3d4f5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_agent_workflows_name")
    op.create_index("ux_agent_workflows_name", "agent_workflows", ["name"], unique=True)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ux_agent_workflows_name")
    op.create_index("ix_agent_workflows_name", "agent_workflows", ["name"], unique=False)
