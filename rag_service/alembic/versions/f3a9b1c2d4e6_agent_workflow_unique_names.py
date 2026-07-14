"""Make agent workflow names unique

Revision ID: f3a9b1c2d4e6
Revises: c8f1a2d3e4b5
Create Date: 2026-07-08 00:00:00.000000

"""
from alembic import op


revision = "f3a9b1c2d4e6"
down_revision = "c8f1a2d3e4b5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_agent_workflows_name")
    op.execute(
        """
        WITH duplicate_names AS (
            SELECT
                id,
                name,
                row_number() OVER (
                    PARTITION BY name
                    ORDER BY
                        is_builtin DESC,
                        CASE visibility
                            WHEN 'public' THEN 0
                            WHEN 'internal' THEN 1
                            WHEN 'deleted' THEN 3
                            ELSE 2
                        END,
                        created_at ASC NULLS LAST,
                        id ASC
                ) AS duplicate_rank
            FROM agent_workflows
            WHERE name IS NOT NULL
        )
        UPDATE agent_workflows AS workflow
        SET
            name = duplicate_names.name || ' (' || workflow.id || ')',
            updated_at = now()
        FROM duplicate_names
        WHERE workflow.id = duplicate_names.id
          AND duplicate_names.duplicate_rank > 1
        """
    )
    op.create_index("ux_agent_workflows_name", "agent_workflows", ["name"], unique=True)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ux_agent_workflows_name")
    op.create_index("ix_agent_workflows_name", "agent_workflows", ["name"], unique=False)
