"""Reset framework-neutral workflow and trace contracts to version 1.

Revision ID: 5c1e8a7d9b3f
Revises: 4b7e2d9a1c5f, k5e2a8c4d7f1
"""

import json
from pathlib import Path

import sqlalchemy as sa
from alembic import op


revision = "5c1e8a7d9b3f"
down_revision = ("4b7e2d9a1c5f", "k5e2a8c4d7f1")
branch_labels = None
depends_on = None

_BUILTIN_DIRECTORY = Path(__file__).resolve().parents[2] / "app" / "agent_workflows" / "builtins"


def _builtin_specs() -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(_BUILTIN_DIRECTORY.glob("*.json"))]


def upgrade() -> None:
    op.execute("DELETE FROM agent_run_events")
    op.execute(
        """
        UPDATE agent_runs
           SET debug_trace_json = NULL,
               runtime_binding_status = 'invalidated',
               run_metadata_json = coalesce(run_metadata_json, '{}'::jsonb) ||
                   jsonb_build_object(
                       'trace_invalidated_reason', 'trace_invalidated_by_contract_reset',
                       'workflow_contract_version', 1
                   ),
               error_json = CASE
                   WHEN status IN ('running', 'awaiting_human') THEN
                       jsonb_build_object(
                           'code', 'workflow_contract_invalidated',
                           'message', 'This run uses an invalidated workflow contract and cannot continue.',
                           'retryable', false
                       )
                   ELSE error_json
               END,
               status = CASE WHEN status IN ('running', 'awaiting_human') THEN 'failed' ELSE status END,
               completed_at = CASE
                   WHEN status IN ('running', 'awaiting_human') THEN coalesce(completed_at, now())
                   ELSE completed_at
               END
        """
    )
    op.execute(
        """
        UPDATE agent_tasks
           SET config_json = coalesce(config_json, '{}'::jsonb) ||
                   jsonb_build_object(
                       'workflow_contract_invalidated', true,
                       'workflow_contract_version', 1
                   ),
               updated_at = now()
        """
    )
    op.execute(
        """
        UPDATE agent_tasks
           SET status = 'failed',
               terminal_reason = 'workflow_contract_invalidated',
               active_run_id = NULL,
               lease_owner = NULL,
               lease_expires_at = NULL,
               completed_at = coalesce(completed_at, now()),
               updated_at = now()
         WHERE status IN ('created', 'queued', 'running', 'pausing', 'paused', 'awaiting_approval', 'cancelling')
        """
    )
    op.execute(
        """
        UPDATE agent_workflows
           SET visibility = 'deleted',
               schema_version = 1,
               validation_result_json = jsonb_build_object(
                   'valid', false,
                   'errors', jsonb_build_array('workflow_contract_invalidated')
               ),
               metadata_json = coalesce(metadata_json, '{}'::jsonb) ||
                   jsonb_build_object('version', 1, 'version_id', concat(id, chr(58), 'v1'), 'contract_invalidated', true),
               updated_at = now()
         WHERE NOT is_builtin
        """
    )
    connection = op.get_bind()
    replace_builtin = sa.text(
        """
        UPDATE agent_workflows
           SET schema_version = 1,
               spec_json = CAST(:spec_json AS jsonb),
               metadata_json = (coalesce(metadata_json, '{}'::jsonb) - 'contract_invalidated') ||
                   jsonb_build_object('version', 1, 'version_id', concat(id, chr(58), 'v1')),
               updated_at = now()
         WHERE id = :workflow_id AND is_builtin
        """
    )
    for spec in _builtin_specs():
        workflow_id = str(spec["builtin_key"])
        workflow_spec = dict(spec["spec_json"])
        connection.execute(
            replace_builtin,
            {"workflow_id": workflow_id, "spec_json": json.dumps(workflow_spec, sort_keys=True)},
        )


def downgrade() -> None:
    raise RuntimeError("The framework-neutral contract reset is intentionally irreversible")
