"""Compatibility marker for removed agent pattern version merge

Revision ID: d4b7c2e9a6f1
Revises: c8f1a2d3e4b5
Create Date: 2026-07-07 00:00:00.000000

This revision existed on the PR branch before agent workflows were simplified.
Some local branch databases may already be stamped at this revision. Keep it in
the graph so Alembic can locate those databases and apply the follow-up cleanup
migration. For databases still at the prior revision with the old split
template/version schema, this performs the original merge idempotently.
"""
from alembic import op


revision = "d4b7c2e9a6f1"
down_revision = "c8f1a2d3e4b5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF to_regclass('public.agent_pattern_templates') IS NOT NULL
               AND NOT EXISTS (
                   SELECT 1
                   FROM information_schema.columns
                   WHERE table_schema = 'public'
                     AND table_name = 'agent_pattern_templates'
                     AND column_name = 'spec_json'
               ) THEN
                ALTER TABLE agent_pattern_templates
                    ADD COLUMN schema_version integer NOT NULL DEFAULT 2,
                    ADD COLUMN spec_json jsonb NOT NULL DEFAULT '{}'::jsonb,
                    ADD COLUMN validation_result_json jsonb NOT NULL DEFAULT '{}'::jsonb,
                    ADD COLUMN metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb;
            END IF;

            IF to_regclass('public.agent_pattern_template_versions') IS NOT NULL THEN
                UPDATE agent_pattern_templates AS t
                SET
                    schema_version = v.schema_version,
                    spec_json = v.spec_json,
                    validation_result_json = v.validation_result_json,
                    metadata_json = jsonb_strip_nulls(
                        COALESCE(t.metadata_json, '{}'::jsonb)
                        || jsonb_build_object(
                            'version', v.version,
                            'version_id', v.id,
                            'changelog', v.changelog,
                            'version_created_at', v.created_at
                        )
                    )
                FROM agent_pattern_template_versions AS v
                WHERE v.id = t.current_version_id;

                UPDATE agent_pattern_templates AS t
                SET
                    schema_version = v.schema_version,
                    spec_json = v.spec_json,
                    validation_result_json = v.validation_result_json,
                    metadata_json = jsonb_strip_nulls(
                        COALESCE(t.metadata_json, '{}'::jsonb)
                        || jsonb_build_object(
                            'version', v.version,
                            'version_id', v.id,
                            'changelog', v.changelog,
                            'version_created_at', v.created_at
                        )
                    )
                FROM (
                    SELECT DISTINCT ON (template_id)
                        template_id,
                        id,
                        version,
                        schema_version,
                        spec_json,
                        validation_result_json,
                        changelog,
                        created_at
                    FROM agent_pattern_template_versions
                    ORDER BY template_id, version DESC
                ) AS v
                WHERE v.template_id = t.id
                  AND t.spec_json = '{}'::jsonb;
            END IF;
        END $$;
        """
    )
    op.execute("DROP INDEX IF EXISTS ix_agent_pattern_template_versions_version")
    op.execute("DROP INDEX IF EXISTS ix_agent_pattern_template_versions_template_id")
    op.execute("DROP INDEX IF EXISTS idx_agent_pattern_template_version_unique")
    op.execute("DROP TABLE IF EXISTS agent_pattern_template_versions")
    op.execute("DROP INDEX IF EXISTS ix_agent_pattern_templates_owner_id")
    op.execute("DROP INDEX IF EXISTS ix_agent_pattern_templates_current_version_id")
    op.execute(
        """
        DO $$
        BEGIN
            IF to_regclass('public.agent_pattern_templates') IS NOT NULL THEN
                ALTER TABLE agent_pattern_templates DROP COLUMN IF EXISTS owner_id;
                ALTER TABLE agent_pattern_templates DROP COLUMN IF EXISTS current_version_id;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    pass
