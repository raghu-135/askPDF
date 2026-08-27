"""Add framework-neutral runtime identity metadata.

Revision ID: 9b4d6e2f1a7c
Revises: a8d3f1c6e4b2
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "9b4d6e2f1a7c"
down_revision = "a8d3f1c6e4b2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_workflows", sa.Column("framework", sa.String(), nullable=False, server_default="langgraph"))
    op.add_column("agent_workflows", sa.Column("builder_id", sa.String(), nullable=False, server_default="langgraph_graph"))
    op.add_column("agent_workflows", sa.Column("category", sa.String(), nullable=True))
    op.create_index("ix_agent_workflows_framework", "agent_workflows", ["framework"], unique=False)
    op.create_index("ix_agent_workflows_builder_id", "agent_workflows", ["builder_id"], unique=False)
    op.create_index("ix_agent_workflows_category", "agent_workflows", ["category"], unique=False)

    op.add_column("agent_runs", sa.Column("framework", sa.String(), nullable=False, server_default="langgraph"))
    op.add_column("agent_runs", sa.Column("builder_id", sa.String(), nullable=False, server_default="langgraph_graph"))
    op.add_column("agent_runs", sa.Column("definition_category", sa.String(), nullable=True))
    op.add_column(
        "agent_runs",
        sa.Column(
            "runtime_binding_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.add_column("agent_runs", sa.Column("runtime_binding_status", sa.String(), nullable=False, server_default="active"))
    op.create_index("ix_agent_runs_framework", "agent_runs", ["framework"], unique=False)
    op.create_index("ix_agent_runs_builder_id", "agent_runs", ["builder_id"], unique=False)
    op.create_index("ix_agent_runs_definition_category", "agent_runs", ["definition_category"], unique=False)

    op.execute(
        """
        UPDATE agent_workflows
        SET framework = COALESCE(NULLIF(metadata_json->>'framework', ''), 'langgraph'),
            builder_id = COALESCE(NULLIF(metadata_json->>'builder_id', ''), 'langgraph_graph'),
            category = COALESCE(
                NULLIF(metadata_json->>'category', ''),
                CASE id
                    WHEN 'router_rag_agent' THEN 'router'
                    WHEN 'plan_execute_rag_agent' THEN 'replanner'
                    WHEN 'evaluator_replanner_rag_agent' THEN 'replanner'
                    WHEN 'orchestrator_worker_rag_agent' THEN 'replanner'
                    WHEN 'corrective_self_rag_agent' THEN 'replanner'
                    WHEN 'deep_research_agent' THEN 'deep'
                    ELSE NULL
                END
            )
        """
    )
    op.execute(
        """
        UPDATE agent_runs AS runs
        SET framework = COALESCE(workflows.framework, 'langgraph'),
            builder_id = COALESCE(workflows.builder_id, 'langgraph_graph'),
            definition_category = workflows.category,
            runtime_binding_status = 'active',
            runtime_binding_json = CASE
                WHEN runs.runtime_binding_json = '{}'::jsonb
                     AND runs.checkpoint_thread_id IS NOT NULL
                THEN jsonb_build_object(
                    'binding_type', 'langgraph_checkpoint',
                    'payload', jsonb_build_object('checkpoint_thread_id', runs.checkpoint_thread_id)
                )
                ELSE runs.runtime_binding_json
            END
        FROM agent_workflows AS workflows
        WHERE runs.workflow_id = workflows.id
        """
    )
    op.execute(
        """
        UPDATE agent_runs
        SET runtime_binding_status = 'legacy_unresolved',
            runtime_binding_json = jsonb_build_object(
                'binding_type', 'legacy_unresolved',
                'payload', jsonb_build_object('workflow_id', workflow_id)
            )
        WHERE NOT EXISTS (SELECT 1 FROM agent_workflows WHERE agent_workflows.id = agent_runs.workflow_id)
        """
    )


def downgrade() -> None:
    op.drop_index("ix_agent_runs_definition_category", table_name="agent_runs")
    op.drop_index("ix_agent_runs_builder_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_framework", table_name="agent_runs")
    op.drop_column("agent_runs", "runtime_binding_status")
    op.drop_column("agent_runs", "runtime_binding_json")
    op.drop_column("agent_runs", "definition_category")
    op.drop_column("agent_runs", "builder_id")
    op.drop_column("agent_runs", "framework")
    op.drop_index("ix_agent_workflows_category", table_name="agent_workflows")
    op.drop_index("ix_agent_workflows_builder_id", table_name="agent_workflows")
    op.drop_index("ix_agent_workflows_framework", table_name="agent_workflows")
    op.drop_column("agent_workflows", "category")
    op.drop_column("agent_workflows", "builder_id")
    op.drop_column("agent_workflows", "framework")
