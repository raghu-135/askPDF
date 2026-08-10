"""Add durable Deep Research task storage.

Revision ID: a8d3f1c6e4b2
Revises: e7c4a1b9d2f6
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "a8d3f1c6e4b2"
down_revision = "e7c4a1b9d2f6"
branch_labels = None
depends_on = None


TASK_STATUSES = "'created','queued','running','pausing','paused','awaiting_approval','cancelling','cancelled','completed','failed','expired'"


def _timestamps(*, updated: bool = False) -> list[sa.Column]:
    columns = [sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())]
    if updated:
        columns.append(sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()))
    return columns


def upgrade() -> None:
    op.create_table(
        "agent_tasks",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("thread_id", sa.String(), sa.ForeignKey("threads.id", ondelete="CASCADE"), nullable=False),
        sa.Column("project_id", sa.String(), sa.ForeignKey("projects.id", ondelete="SET NULL"), nullable=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("workflow_id", sa.String(), sa.ForeignKey("agent_workflows.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("objective_hash", sa.String(), nullable=False),
        sa.Column("create_idempotency_key", sa.String(), nullable=False),
        sa.Column("config_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("status", sa.String(), nullable=False, server_default="created"),
        sa.Column("primary_run_id", sa.String(), nullable=True),
        sa.Column("active_run_id", sa.String(), nullable=True),
        sa.Column("latest_run_attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("lease_owner", sa.String(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_todos", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_todos", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("current_phase", sa.String(), nullable=False, server_default="created"),
        sa.Column("terminal_reason", sa.String(), nullable=True),
        sa.Column("budgets_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("paused_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deletion_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deletion_completed_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(updated=True),
        sa.CheckConstraint("length(btrim(objective)) > 0", name="ck_agent_tasks_objective_nonempty"),
        sa.CheckConstraint("length(btrim(objective_hash)) > 0", name="ck_agent_tasks_objective_hash_nonempty"),
        sa.CheckConstraint("length(btrim(create_idempotency_key)) > 0", name="ck_agent_tasks_idempotency_nonempty"),
        sa.CheckConstraint(f"status in ({TASK_STATUSES})", name="ck_agent_tasks_status"),
        sa.CheckConstraint("version >= 1 and latest_run_attempt >= 0", name="ck_agent_tasks_versions"),
        sa.CheckConstraint("progress between 0 and 100 and completed_todos >= 0 and total_todos >= 0", name="ck_agent_tasks_progress"),
    )
    op.create_index("idx_agent_tasks_thread_created", "agent_tasks", ["thread_id", "created_at"])
    op.create_index("idx_agent_tasks_claim", "agent_tasks", ["status", "lease_expires_at", "queued_at"])
    op.create_index(
        "uq_agent_tasks_owner_idempotency_nullsafe",
        "agent_tasks",
        ["thread_id", sa.text("coalesce(user_id, '')"), "create_idempotency_key"],
        unique=True,
    )
    for column in ("thread_id", "project_id", "user_id", "workflow_id", "primary_run_id", "active_run_id", "lease_owner", "expires_at", "deletion_requested_at"):
        op.create_index(f"ix_agent_tasks_{column}", "agent_tasks", [column])

    op.add_column("agent_runs", sa.Column("task_id", sa.String(), nullable=True))
    op.add_column("agent_runs", sa.Column("parent_run_id", sa.String(), nullable=True))
    op.add_column("agent_runs", sa.Column("task_attempt", sa.Integer(), nullable=False, server_default="1"))
    op.create_foreign_key("fk_agent_runs_task_id", "agent_runs", "agent_tasks", ["task_id"], ["id"], ondelete="CASCADE")
    op.create_foreign_key("fk_agent_runs_parent_run_id", "agent_runs", "agent_runs", ["parent_run_id"], ["id"], ondelete="RESTRICT")
    op.create_check_constraint("ck_agent_runs_task_attempt", "agent_runs", "task_attempt >= 1")
    op.create_index("ix_agent_runs_task_id", "agent_runs", ["task_id"])
    op.create_index("ix_agent_runs_parent_run_id", "agent_runs", ["parent_run_id"])
    op.create_index(
        "uq_agent_runs_one_active_task_run",
        "agent_runs",
        ["task_id"],
        unique=True,
        postgresql_where=sa.text("task_id is not null and status in ('running','awaiting_human')"),
    )

    op.create_table(
        "agent_task_plan_revisions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("task_id", sa.String(), sa.ForeignKey("agent_tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("agent_run_id", sa.String(), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("planner_visit", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("reason", sa.String(), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("completion_criteria_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("ordered_todo_ids_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("plan_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("provenance_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("content_hash", sa.String(), nullable=False),
        *_timestamps(),
        sa.CheckConstraint("revision >= 1 and planner_visit >= 1", name="ck_agent_task_plan_revision_values"),
        sa.UniqueConstraint("task_id", "revision", name="uq_agent_task_plan_revision"),
    )
    op.create_index("ix_agent_task_plan_revisions_task_id", "agent_task_plan_revisions", ["task_id"])
    op.create_index("ix_agent_task_plan_revisions_agent_run_id", "agent_task_plan_revisions", ["agent_run_id"])

    op.create_table(
        "agent_task_todos",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("task_id", sa.String(), sa.ForeignKey("agent_tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("completion_criteria", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="50"),
        sa.Column("required", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("dependency_ids_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("profile_id", sa.String(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("terminal_reason", sa.String(), nullable=True),
        sa.Column("current_subagent_run_id", sa.String(), nullable=True),
        sa.Column("evidence_ids_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("artifact_ids_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("updated_revision", sa.Integer(), nullable=False, server_default="1"),
        *_timestamps(updated=True),
        sa.PrimaryKeyConstraint("task_id", "id"),
        sa.CheckConstraint("status in ('pending','ready','running','blocked','completed','failed','skipped','cancelled')", name="ck_agent_task_todos_status"),
        sa.CheckConstraint("priority between 0 and 100 and progress between 0 and 100", name="ck_agent_task_todos_ranges"),
        sa.CheckConstraint("attempt >= 0 and max_attempts between 1 and 10", name="ck_agent_task_todos_attempts"),
        sa.CheckConstraint("version >= 1 and created_revision >= 1 and updated_revision >= created_revision", name="ck_agent_task_todos_versions"),
    )
    op.create_index("idx_agent_task_todos_schedule", "agent_task_todos", ["task_id", "status", "priority"])

    op.create_table(
        "agent_task_subagent_runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("task_id", sa.String(), sa.ForeignKey("agent_tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("agent_run_id", sa.String(), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("todo_id", sa.String(), nullable=False),
        sa.Column("execution_key", sa.String(), nullable=False, unique=True),
        sa.Column("profile_id", sa.String(), nullable=False),
        sa.Column("plan_revision", sa.Integer(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("usage_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("tool_policy_hash", sa.String(), nullable=False),
        sa.Column("timeout_ms", sa.Integer(), nullable=False),
        sa.Column("output_artifact_ids_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("error_json", postgresql.JSONB(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(["task_id", "todo_id"], ["agent_task_todos.task_id", "agent_task_todos.id"], ondelete="CASCADE"),
        sa.CheckConstraint("status in ('queued','running','completed','failed','timed_out','cancelled')", name="ck_agent_task_subagent_status"),
        sa.CheckConstraint("attempt >= 1 and plan_revision >= 1 and timeout_ms > 0", name="ck_agent_task_subagent_values"),
    )
    op.create_index("ix_agent_task_subagent_runs_task_id", "agent_task_subagent_runs", ["task_id"])
    op.create_index("ix_agent_task_subagent_runs_agent_run_id", "agent_task_subagent_runs", ["agent_run_id"])
    op.create_index("ix_agent_task_subagent_runs_todo_id", "agent_task_subagent_runs", ["todo_id"])
    op.create_index("ix_agent_task_subagent_runs_status", "agent_task_subagent_runs", ["status"])

    op.create_table(
        "agent_task_artifacts",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("task_id", sa.String(), sa.ForeignKey("agent_tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("agent_run_id", sa.String(), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("todo_id", sa.String(), nullable=True),
        sa.Column("subagent_run_id", sa.String(), sa.ForeignKey("agent_task_subagent_runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("ownership_key", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("object_key", sa.String(), nullable=False, unique=True),
        sa.Column("media_type", sa.String(), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("provenance_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("source_refs_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("summary_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("supersedes_id", sa.String(), sa.ForeignKey("agent_task_artifacts.id", ondelete="SET NULL"), nullable=True),
        sa.Column("validity", sa.String(), nullable=False, server_default="valid"),
        sa.Column("sensitivity", sa.String(), nullable=False, server_default="private"),
        sa.Column("retention_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.CheckConstraint("kind in ('tool_output','intermediate_report','context_summary','final_report')", name="ck_agent_task_artifacts_kind"),
        sa.CheckConstraint("validity in ('valid','invalid','deleted') and sensitivity in ('private','sensitive')", name="ck_agent_task_artifacts_state"),
        sa.CheckConstraint("byte_size >= 0 and version >= 1", name="ck_agent_task_artifacts_values"),
        sa.CheckConstraint("length(btrim(ownership_key)) > 0", name="ck_agent_task_artifacts_ownership_key"),
        sa.UniqueConstraint("agent_run_id", "ownership_key", "sha256", "kind", name="uq_agent_task_artifact_content"),
    )
    op.create_index(
        "uq_agent_task_artifacts_final_report",
        "agent_task_artifacts",
        ["agent_run_id"],
        unique=True,
        postgresql_where=sa.text("kind = 'final_report' and validity = 'valid' and deleted_at is null"),
    )
    for column in ("task_id", "agent_run_id", "todo_id", "subagent_run_id", "ownership_key", "kind", "validity", "retention_until"):
        op.create_index(f"ix_agent_task_artifacts_{column}", "agent_task_artifacts", [column])

    op.create_table(
        "agent_task_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("task_id", sa.String(), sa.ForeignKey("agent_tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("actor_type", sa.String(), nullable=False),
        sa.Column("actor_id", sa.String(), nullable=True),
        sa.Column("agent_run_id", sa.String(), nullable=True),
        sa.Column("todo_id", sa.String(), nullable=True),
        sa.Column("subagent_run_id", sa.String(), nullable=True),
        sa.Column("artifact_id", sa.String(), nullable=True),
        sa.Column("payload_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("policy_hash", sa.String(), nullable=True),
        sa.Column("config_hash", sa.String(), nullable=True),
        *_timestamps(),
        sa.CheckConstraint("sequence >= 1", name="ck_agent_task_events_sequence"),
        sa.UniqueConstraint("task_id", "sequence", name="uq_agent_task_event_sequence"),
    )
    op.create_index("idx_agent_task_events_stream", "agent_task_events", ["task_id", "sequence"])
    op.create_index("idx_agent_task_events_run_stream", "agent_task_events", ["task_id", "agent_run_id", "sequence"])
    op.create_index("ix_agent_task_events_event_type", "agent_task_events", ["event_type"])

    op.create_table(
        "agent_task_commands",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("task_id", sa.String(), sa.ForeignKey("agent_tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("expected_version", sa.Integer(), nullable=False),
        sa.Column("actor_id", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="accepted"),
        sa.Column("result_version", sa.Integer(), nullable=True),
        sa.Column("result_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.CheckConstraint("action in ('start','pause','resume','cancel','retry','expire','delete')", name="ck_agent_task_commands_action"),
        sa.CheckConstraint("status in ('accepted','completed','rejected') and expected_version >= 1", name="ck_agent_task_commands_state"),
        sa.UniqueConstraint("task_id", "action", "idempotency_key", name="uq_agent_task_command_idempotency"),
    )
    op.create_index("ix_agent_task_commands_task_id", "agent_task_commands", ["task_id"])


def downgrade() -> None:
    op.drop_table("agent_task_commands")
    op.drop_table("agent_task_events")
    op.drop_table("agent_task_artifacts")
    op.drop_table("agent_task_subagent_runs")
    op.drop_table("agent_task_todos")
    op.drop_table("agent_task_plan_revisions")
    op.drop_index("uq_agent_runs_one_active_task_run", table_name="agent_runs")
    op.drop_index("ix_agent_runs_parent_run_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_task_id", table_name="agent_runs")
    op.drop_constraint("ck_agent_runs_task_attempt", "agent_runs", type_="check")
    op.drop_constraint("fk_agent_runs_parent_run_id", "agent_runs", type_="foreignkey")
    op.drop_constraint("fk_agent_runs_task_id", "agent_runs", type_="foreignkey")
    op.drop_column("agent_runs", "task_attempt")
    op.drop_column("agent_runs", "parent_run_id")
    op.drop_column("agent_runs", "task_id")
    op.drop_table("agent_tasks")
