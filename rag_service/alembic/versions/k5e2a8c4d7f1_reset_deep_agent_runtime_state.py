"""Reset disposable Deep Agents state and remove obsolete version columns.

Revision ID: k5e2a8c4d7f1
Revises: j4d9e6f1b3c5

This feature-branch migration intentionally deletes Deep Agents execution
state. Its downgrade recreates schema only; deleted data is not recoverable.
"""

from alembic import op


revision = "k5e2a8c4d7f1"
down_revision = "j4d9e6f1b3c5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF to_regclass('public.agent_tasks') IS NOT NULL THEN
                DELETE FROM agent_tasks;
            END IF;

            IF to_regclass('public.agent_runs') IS NOT NULL THEN
                DELETE FROM agent_runs
                WHERE definition_category = 'deep'
                   OR workflow_id IN ('deep_research_agent', 'hermes_rag_agent');
                ALTER TABLE agent_runs DROP COLUMN IF EXISTS runtime_binding_version;
            END IF;

            IF to_regclass('public.runtime_executions') IS NOT NULL THEN
                DELETE FROM runtime_executions;
            END IF;

            IF to_regclass('public.runtime_events') IS NOT NULL THEN
                ALTER TABLE runtime_events
                    DROP COLUMN IF EXISTS runtime_version,
                    DROP COLUMN IF EXISTS contract_version;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF to_regclass('public.agent_runs') IS NOT NULL THEN
                ALTER TABLE agent_runs
                    ADD COLUMN IF NOT EXISTS runtime_binding_version integer NOT NULL DEFAULT 1;
            END IF;

            IF to_regclass('public.runtime_events') IS NOT NULL THEN
                ALTER TABLE runtime_events
                    ADD COLUMN IF NOT EXISTS runtime_version text,
                    ADD COLUMN IF NOT EXISTS contract_version integer;
            END IF;
        END $$;
        """
    )
