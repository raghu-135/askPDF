"""Merge agent pattern template versions into templates

Revision ID: d4b7c2e9a6f1
Revises: c8f1a2d3e4b5
Create Date: 2026-07-07 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "d4b7c2e9a6f1"
down_revision = "c8f1a2d3e4b5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_pattern_templates", sa.Column("schema_version", sa.Integer(), nullable=False, server_default="2"))
    op.add_column(
        "agent_pattern_templates",
        sa.Column("spec_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.add_column(
        "agent_pattern_templates",
        sa.Column("validation_result_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.add_column(
        "agent_pattern_templates",
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )

    op.execute(
        """
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
        WHERE v.id = t.current_version_id
        """
    )
    op.execute(
        """
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
          AND t.spec_json = '{}'::jsonb
        """
    )

    op.drop_index(op.f("ix_agent_pattern_template_versions_version"), table_name="agent_pattern_template_versions")
    op.drop_index(op.f("ix_agent_pattern_template_versions_template_id"), table_name="agent_pattern_template_versions")
    op.drop_index("idx_agent_pattern_template_version_unique", table_name="agent_pattern_template_versions")
    op.drop_table("agent_pattern_template_versions")

    op.drop_index(op.f("ix_agent_pattern_templates_owner_id"), table_name="agent_pattern_templates")
    op.drop_index(op.f("ix_agent_pattern_templates_current_version_id"), table_name="agent_pattern_templates")
    op.drop_column("agent_pattern_templates", "owner_id")
    op.drop_column("agent_pattern_templates", "current_version_id")


def downgrade() -> None:
    op.add_column("agent_pattern_templates", sa.Column("current_version_id", sa.String(), nullable=True))
    op.add_column("agent_pattern_templates", sa.Column("owner_id", sa.String(), nullable=True))
    op.create_index(op.f("ix_agent_pattern_templates_current_version_id"), "agent_pattern_templates", ["current_version_id"], unique=False)
    op.create_index(op.f("ix_agent_pattern_templates_owner_id"), "agent_pattern_templates", ["owner_id"], unique=False)

    op.create_table(
        "agent_pattern_template_versions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("template_id", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("spec_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("validation_result_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("changelog", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["template_id"], ["agent_pattern_templates.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_agent_pattern_template_version_unique", "agent_pattern_template_versions", ["template_id", "version"], unique=True)
    op.create_index(op.f("ix_agent_pattern_template_versions_template_id"), "agent_pattern_template_versions", ["template_id"], unique=False)
    op.create_index(op.f("ix_agent_pattern_template_versions_version"), "agent_pattern_template_versions", ["version"], unique=False)

    op.execute(
        """
        INSERT INTO agent_pattern_template_versions (
            id,
            template_id,
            version,
            schema_version,
            spec_json,
            validation_result_json,
            changelog,
            created_at
        )
        SELECT
            COALESCE(metadata_json->>'version_id', id || ':v' || COALESCE((metadata_json->>'version'), '1')),
            id,
            COALESCE((metadata_json->>'version')::integer, 1),
            schema_version,
            spec_json,
            validation_result_json,
            metadata_json->>'changelog',
            COALESCE((metadata_json->>'version_created_at')::timestamptz, created_at)
        FROM agent_pattern_templates
        """
    )
    op.execute(
        """
        UPDATE agent_pattern_templates
        SET current_version_id = COALESCE(metadata_json->>'version_id', id || ':v' || COALESCE((metadata_json->>'version'), '1'))
        """
    )

    op.drop_column("agent_pattern_templates", "metadata_json")
    op.drop_column("agent_pattern_templates", "validation_result_json")
    op.drop_column("agent_pattern_templates", "spec_json")
    op.drop_column("agent_pattern_templates", "schema_version")
