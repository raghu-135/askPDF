"""Add explicit runtime projection recovery state and lineage acknowledgement."""

from alembic import op
import sqlalchemy as sa


revision = "c3e9f5a7b2d4"
down_revision = "b2d8e4f6a1c3"
branch_labels = None
depends_on = None


TASK_STATUSES = (
    "'created','queued','running','pausing','paused','awaiting_approval',"
    "'cancelling','recovery_required','cancelled','completed','failed','expired'"
)


def upgrade() -> None:
    op.drop_constraint("ck_agent_tasks_status", "agent_tasks", type_="check")
    op.create_check_constraint("ck_agent_tasks_status", "agent_tasks", f"status in ({TASK_STATUSES})")
    op.add_column(
        "agent_task_runtime_deltas",
        sa.Column("applied_runtime_plan_revision", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_check_constraint(
        "ck_agent_task_runtime_delta_runtime_revision",
        "agent_task_runtime_deltas",
        "applied_runtime_plan_revision >= 0",
    )
    op.execute(sa.text("""
        update agent_tasks
        set status = 'recovery_required'
        where current_phase = 'runtime_projection_recovery_required'
          and status not in ('cancelled','completed','failed','expired')
    """))
    op.execute(sa.text("""
        update agent_runs r
        set status = 'recovery_required'
        from agent_tasks t
        where t.active_run_id = r.id and t.status = 'recovery_required'
          and r.status in ('running','awaiting_human')
    """))
    op.execute(sa.text("""
        update agent_task_commands c
        set status = 'accepted', completed_at = null,
            result_json = jsonb_set(c.result_json, '{delivery_state}', '"runtime_applied"'::jsonb, true)
        from agent_tasks t
        where c.task_id = t.id and t.status = 'recovery_required'
          and c.action = 'steer' and c.status = 'completed'
          and c.result_json->>'delivery_state' = 'applied'
    """))


def downgrade() -> None:
    op.execute("update agent_runs set status = 'failed' where status = 'recovery_required'")
    op.execute("update agent_tasks set status = 'failed' where status = 'recovery_required'")
    op.drop_constraint(
        "ck_agent_task_runtime_delta_runtime_revision", "agent_task_runtime_deltas", type_="check"
    )
    op.drop_column("agent_task_runtime_deltas", "applied_runtime_plan_revision")
    op.drop_constraint("ck_agent_tasks_status", "agent_tasks", type_="check")
    op.create_check_constraint(
        "ck_agent_tasks_status",
        "agent_tasks",
        "status in ('created','queued','running','pausing','paused','awaiting_approval','cancelling','cancelled','completed','failed','expired')",
    )
