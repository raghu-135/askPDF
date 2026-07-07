"""Rename agent pattern storage to workflows

Revision ID: e9a1b2c3d4f5
Revises: d4b7c2e9a6f1
Create Date: 2026-07-07 13:45:00.000000

"""
from alembic import op


revision = "e9a1b2c3d4f5"
down_revision = "d4b7c2e9a6f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF to_regclass('public.agent_workflows') IS NULL
               AND to_regclass('public.agent_pattern_templates') IS NOT NULL THEN
                ALTER TABLE agent_pattern_templates RENAME TO agent_workflows;
            END IF;

            IF EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'agent_runs'
                  AND column_name = 'template_id'
            ) AND NOT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'agent_runs'
                  AND column_name = 'workflow_id'
            ) THEN
                ALTER TABLE agent_runs RENAME COLUMN template_id TO workflow_id;
            END IF;
        END $$;
        """
    )
    op.execute("DROP INDEX IF EXISTS idx_agent_pattern_template_builtin")
    op.execute("DROP INDEX IF EXISTS ix_agent_pattern_templates_name")
    op.execute("DROP INDEX IF EXISTS ix_agent_pattern_templates_visibility")
    op.execute("DROP INDEX IF EXISTS ix_agent_runs_template_id")
    op.execute("CREATE INDEX IF NOT EXISTS idx_agent_workflow_builtin ON agent_workflows (is_builtin)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_workflows_name ON agent_workflows (name)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_workflows_visibility ON agent_workflows (visibility)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_runs_workflow_id ON agent_runs (workflow_id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_agent_runs_workflow_id")
    op.execute("DROP INDEX IF EXISTS ix_agent_workflows_visibility")
    op.execute("DROP INDEX IF EXISTS ix_agent_workflows_name")
    op.execute("DROP INDEX IF EXISTS idx_agent_workflow_builtin")
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_runs_template_id ON agent_runs (workflow_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_pattern_templates_visibility ON agent_workflows (visibility)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_pattern_templates_name ON agent_workflows (name)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_agent_pattern_template_builtin ON agent_workflows (is_builtin)")
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'agent_runs'
                  AND column_name = 'workflow_id'
            ) AND NOT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'agent_runs'
                  AND column_name = 'template_id'
            ) THEN
                ALTER TABLE agent_runs RENAME COLUMN workflow_id TO template_id;
            END IF;

            IF to_regclass('public.agent_pattern_templates') IS NULL
               AND to_regclass('public.agent_workflows') IS NOT NULL THEN
                ALTER TABLE agent_workflows RENAME TO agent_pattern_templates;
            END IF;
        END $$;
        """
    )
